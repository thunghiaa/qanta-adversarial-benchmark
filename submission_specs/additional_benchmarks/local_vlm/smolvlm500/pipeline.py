"""Custom Quizbowl pipelines for QANTA 2026 Arena.
Registers `quizbowl-tossup` and `quizbowl-bonus`. The build script rewrites the
two flags below per model (LLM vs VLM). Every name passed to register_pipeline
is imported from transformers, per the submission requirements.

Prompts here are adapted from the strongest hand-tuned submissions in
advcal-requests / the gpt-4o-mini teammate cache: an elite pyramidal-aware
tossup answerer with a confidence ladder, and the elite BONUS specialist prompt.
"""
import json, re
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import (
    Pipeline,
    AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor,
)

PT_MODEL = AutoModelForImageTextToText   # build script -> AutoModelForImageTextToText for VLMs
IS_VLM = True                    # build script -> True for VLMs


def _to_pil(images):
    from PIL import Image
    out = []
    for im in images or []:
        out.append(Image.open(im).convert("RGB") if isinstance(im, str) else im)
    return out


def _extract(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # strip reasoning traces
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    # Fallback: model emitted prose, not JSON. Recover an answer so a model that
    # replies "ANSWER: Leo Tolstoy" or just "Leo Tolstoy" still yields a usable field.
    m2 = re.search(r'answer["\']?\s*[:=]\s*["\']?([^\n"\']+)', text, re.I)
    if m2:
        return {"answer": m2.group(1).strip()[:200]}
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return {"answer": first[:200]}


class _Base(Pipeline):
    # The tossup eval calls the pipeline once per run_index (~20 prefixes) on every
    # one of ~225 questions -> ~4.5k generations. With 200 new tokens that overruns
    # the server's per-job time budget (every 7B+ model timed out). The output is a
    # short JSON, so a tight cap is the single biggest speedup. Bonus is ~225 calls
    # total, so it can afford a few more tokens for the explanation.
    MAX_NEW = 96

    def _sanitize_parameters(self, **kw):
        return {}, {}, {}

    def preprocess(self, inputs):
        return inputs

    def _forward(self, inputs):
        revealed = inputs.get("question_text") or inputs.get("part") or ""
        n = len(revealed.split())
        if self._skip_generation(n):
            # Below the word count at which a buzz could ever clear the gate, so the
            # answer here can never be scored. Skip the (expensive) generation
            # entirely -- this is what lets a 7-8B model finish all the run_indices
            # of ~225 questions inside the per-job time budget instead of timing out.
            return {"text": "", "n_words": n, "skipped": True}
        text = self._gen(self._prompt(inputs), inputs.get("images"))
        return {"text": text, "n_words": n, "skipped": False}

    def _skip_generation(self, n_words):
        return False  # bonus + base: always generate; tossup overrides this

    def _proc(self):
        if not hasattr(self, "_p"):
            self._p = AutoProcessor.from_pretrained(self.model.name_or_path, trust_remote_code=True)
        return self._p

    def _gen(self, prompt, images=None):
        if IS_VLM and images:
            proc = self._proc()
            msgs = [{"role": "user", "content": [{"type": "image"} for _ in images] +
                     [{"type": "text", "text": prompt}]}]
            chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=chat, images=_to_pil(images), return_tensors="pt").to(
                self.model.device, self.model.dtype)  # BatchFeature.to casts float tensors only
            ids = self.model.generate(**inp, max_new_tokens=self.MAX_NEW, do_sample=False)
            return proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            # enable_thinking=False keeps Qwen3-family models out of <think> mode, which
            # would otherwise spend the whole MAX_NEW budget reasoning and never emit JSON.
            # Older templates don't take the kwarg -> fall back.
            try:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inp = tok(text, return_tensors="pt").to(self.model.device)
        ids = self.model.generate(**inp, max_new_tokens=self.MAX_NEW, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        return tok.decode(ids[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)


class QuizbowlTossupPipeline(_Base):
    MAX_NEW = 48          # tossup answer JSON is tiny; this is the ~4.5k-call hot path
    BUZZ_FULL_WORDS = 50  # by ~this many revealed words, trust the model's confidence fully
    BUZZ_THRESHOLD = 0.6  # buzz only when confidence * completeness clears this bar

    @property
    def MIN_BUZZ_WORDS(self):
        # A buzz needs confidence * completeness >= THRESHOLD with confidence <= 1, so it
        # is impossible until completeness >= THRESHOLD, i.e. this many words revealed.
        # Below it we skip generation outright (see _Base._forward).
        return int(self.BUZZ_THRESHOLD * self.BUZZ_FULL_WORDS)

    def _skip_generation(self, n_words):
        return n_words < self.MIN_BUZZ_WORDS

    def _prompt(self, inputs):
        # Adapted from the top hand-tuned tossup submission (Amanvir/gpt-tiger-calib):
        # pyramidal-structure awareness + an explicit confidence ladder + the
        # "For 10 points" giveaway rule + canonical-answer formatting + a worked example.
        return (
            "You are an elite quizbowl player with deep knowledge across history, literature, "
            "science, the arts, religion, mythology, philosophy, social science, and current events.\n"
            "You are reading a TOSSUP revealed progressively. Quizbowl questions are PYRAMIDAL: the "
            "first clues are obscure and uniquely identifying, later clues get easier, and the final "
            'sentence (often starting "For 10 points") is the giveaway. Tie the specific clues to the '
            "one answer they uniquely identify; reason from what you know for certain.\n"
            "Formatting: give the canonical answer. Literature -> full author name; science -> complete "
            "technical term; people -> full name.\n"
            "Confidence (0.00-1.00, two decimals):\n"
            "- 0.00-0.20 wild guess, clues fit many answers\n"
            "- 0.20-0.40 educated guess, real doubt remains\n"
            "- 0.40-0.60 moderate; a few clues point here but rivals exist\n"
            "- 0.60-0.80 strong; most clues converge, few rivals\n"
            "- 0.80-1.00 near-certain; clues are unique to this answer\n"
            '- If you see "For 10 points", you are at the giveaway: answer with confidence >= 0.95.\n'
            "Example\n"
            'QUESTION: "This author opened a novel about an adulterous noblewoman with \'Happy families '
            "are all alike.' He also wrote a sprawling novel set during Napoleon's invasion of Russia.\"\n"
            '{"answer": "Leo Tolstoy", "confidence": 0.93}\n'
            'Respond with ONLY this JSON: {"answer": "<canonical answer>", "confidence": <0.00-1.00>}\n'
            "QUESTION: " + inputs["question_text"])

    def postprocess(self, mo):
        # The model self-reports a flat ~0.95 confidence even on obscure early clues, so
        # trusting it raw means a wrong buzz at token ~11 (-5 pts). Instead gate on
        # confidence scaled by how much of the question is revealed: early clues are hard,
        # so demand more text before buzzing. (Mirrors the word-count calibrators the top
        # competitors use: scaling = 1 + 30/(words+1); conf /= scaling.)
        if mo.get("skipped"):
            return {"answer": "", "confidence": 0.0, "buzz": False}
        d = _extract(mo["text"])
        c = min(max(float(d.get("confidence", 0.5) or 0.5), 0.0), 1.0)
        completeness = min(1.0, mo.get("n_words", 0) / self.BUZZ_FULL_WORDS)
        buzz = (c * completeness) >= self.BUZZ_THRESHOLD
        return {"answer": str(d.get("answer", "")).strip(), "confidence": c, "buzz": buzz}


class QuizbowlBonusPipeline(_Base):
    def _prompt(self, inputs):
        # The exact "elite Quizbowl BONUS specialist" prompt the gpt-4o-mini teammate uses
        # (from the llm cache): confidence scale tied to +10/0 scoring + worked examples +
        # strict JSON + <=30-word explanation citing a clue. Same prompt, bigger model.
        return (
            "You are an elite quizbowl BONUS specialist. You are given a leadin (topic intro) and "
            "one bonus PART that needs a single answer. Find the best answer, then output ONLY JSON.\n"
            "Confidence scale (+10 / 0 scoring):\n"
            "- 0.25 educated guess, several rivals remain\n"
            "- 0.55 likely correct but not iron-clad\n"
            "- 0.80 strong unique clue, few rivals\n"
            "- 0.93 iron-clad, no plausible rivals\n"
            "Example\n"
            'LEADIN: "Bacterium discovered by Koch that causes tuberculosis."\n'
            'PART: "Name this bacterium."\n'
            '{"answer": "Mycobacterium tuberculosis", "confidence": 0.90, '
            '"explanation": "Koch\'s discovery and tuberculosis uniquely indicate M. tuberculosis."}\n'
            "Rules: never output reasoning outside the JSON; explanation <= 30 words; canonical full names.\n"
            'Respond ONLY as JSON: {"answer": "<concise>", "confidence": <0.00-1.00>, '
            '"explanation": "<= 30 words citing a clue>"}\n'
            "LEADIN: " + inputs.get("leadin", "") + "\nPART: " + inputs["part"])

    def postprocess(self, mo):
        d = _extract(mo["text"])
        c = min(max(float(d.get("confidence", 0.5) or 0.5), 0.0), 1.0)
        exp = " ".join(str(d.get("explanation", "")).split()[:30])
        return {"answer": str(d.get("answer", "")).strip(), "confidence": c, "explanation": exp}


PIPELINE_REGISTRY.register_pipeline("quizbowl-tossup", pipeline_class=QuizbowlTossupPipeline, pt_model=PT_MODEL)
PIPELINE_REGISTRY.register_pipeline("quizbowl-bonus",  pipeline_class=QuizbowlBonusPipeline,  pt_model=PT_MODEL)
