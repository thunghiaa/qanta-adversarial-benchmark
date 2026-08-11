"""
Run Hugging Face `transformers` quizbowl custom pipelines for evaluation.

Submission JSON uses `submission_type: hf_pipeline` with `model_name` as the repo
name segment; the full Hub id is `{username}/{model_name}` (same convention as
workflow submission display names).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# Reduce CUDA fragmentation when set before first GPU alloc (best-effort).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import Dataset
from loguru import logger

from shared.workflows.runners import run_and_eval_bonus_dataset, run_and_eval_tossup_dataset


def _ensure_pipeline_registry_compat() -> None:
    """Re-expose ``PIPELINE_REGISTRY`` on ``transformers`` for Hub custom pipeline code.

    Older submission repos use ``from transformers import PIPELINE_REGISTRY``; newer
    transformers releases only define it under ``transformers.pipelines``.
    """
    import transformers

    if "PIPELINE_REGISTRY" not in transformers._class_to_module:
        transformers._class_to_module["PIPELINE_REGISTRY"] = "pipelines"
    if "PIPELINE_REGISTRY" not in transformers.__all__:
        transformers.__all__.append("PIPELINE_REGISTRY")

    try:
        from transformers import PIPELINE_REGISTRY  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Failed to register transformers.PIPELINE_REGISTRY for legacy Hub pipelines"
        ) from exc


def _patch_cached_hub_pipeline(model_id: str) -> None:
    """Rewrite cached Hub ``pipeline.py`` files that import PIPELINE_REGISTRY from transformers."""
    if os.path.isdir(model_id):
        return  # local directory: its pipeline.py is loaded in place, not from the Hub cache
    hf_home = os.environ.get("HF_HOME", "").strip()
    if not hf_home:
        return
    org, _, name = model_id.partition("/")
    if not name:
        return
    safe_name = name.replace("-", "_hyphen_")
    root = Path(hf_home) / "modules" / "transformers_modules" / org / safe_name
    if not root.is_dir():
        return

    for pipeline_py in root.glob("*/pipeline.py"):
        text = pipeline_py.read_text(encoding="utf-8")
        if "from transformers.pipelines import PIPELINE_REGISTRY" in text:
            continue
        if "PIPELINE_REGISTRY" not in text:
            continue
        patched = re.sub(r",\s*PIPELINE_REGISTRY\b", "", text)
        patched = re.sub(r"\bPIPELINE_REGISTRY\s*,\s*", "", patched)
        patched = re.sub(r"\(\s*PIPELINE_REGISTRY\s*\)", "()", patched)
        if patched == text:
            continue
        patched = "from transformers.pipelines import PIPELINE_REGISTRY\n" + patched
        pipeline_py.write_text(patched, encoding="utf-8")
        logger.info(f"Patched legacy PIPELINE_REGISTRY import in {pipeline_py}")


_ensure_pipeline_registry_compat()


def _summarize_images(images: list) -> list[str]:
    """Short descriptions for multimodal debug logs (paths or PIL metadata)."""
    out: list[str] = []
    for img in images:
        if isinstance(img, str):
            out.append(img if len(img) <= 120 else f"{img[:117]}...")
        else:
            size = getattr(img, "size", None)
            out.append(f"<{type(img).__name__} size={size}>" if size else f"<{type(img).__name__}>")
    return out


class _HfTossupPipelineAgent:
    """Mimics `QuizBowlTossupAgent.run` for progressive reveals + early buzz stop."""

    def __init__(self, pipe: Any):
        self.pipe = pipe

    def _single_run(self, question_run: str | dict, run_idx: int) -> dict:
        if isinstance(question_run, str):
            text = question_run
            images: list = []
        else:
            text = question_run.get("text", "")
            images = list(question_run.get("images") or [])
        qid = question_run.get("qid", "") if isinstance(question_run, dict) else ""
        if images:
            logger.info(
                f"[Multimodal Debug] HF tossup run {run_idx} qid={qid}: "
                f"calling pipeline with {len(images)} image(s): {_summarize_images(images)}"
            )
        else:
            logger.info(
                f"[Multimodal Debug] HF tossup run {run_idx} qid={qid}: "
                "calling pipeline with no images (text-only)"
            )
        out = self.pipe({"question_text": text, "images": images})
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if not isinstance(out, dict):
            raise TypeError(f"HF tossup pipeline must return a dict, got {type(out)}")
        guess = out.get("answer")
        guess = "" if guess is None else str(guess)
        confidence = float(out.get("confidence", 0.0))
        buzz = bool(out.get("buzz", False))
        return {
            "run_idx": run_idx,
            "guess": guess,
            "confidence": confidence,
            "logprob": None,
            "buzz": buzz,
            "question_fragment": text,
            "step_contents": [],
            "step_outputs": {},
            "response_time": 0.0,
        }

    def run(self, question_runs: list, early_stop: bool = True):
        for i, question_run in enumerate(question_runs):
            result = self._single_run(question_run, i + 1)
            yield result
            if early_stop and result["buzz"]:
                if i + 1 < len(question_runs):
                    yield self._single_run(question_runs[-1], len(question_runs))
                return


class _HfBonusPipelineAgent:
    """Mimics `QuizBowlBonusAgent.run` (multimodal leadin + part, single pipeline call)."""

    def __init__(self, pipe: Any):
        self.pipe = pipe

    def run(self, leadin: str | dict, part: str | dict) -> dict:
        leadin_text = leadin if isinstance(leadin, str) else leadin.get("text", "")
        part_text = part if isinstance(part, str) else part.get("text", "")
        leadin_images = [] if isinstance(leadin, str) else list(leadin.get("images") or [])
        part_images = [] if isinstance(part, str) else list(part.get("images") or [])
        images = leadin_images + part_images
        qid = ""
        if isinstance(leadin, dict):
            qid = leadin.get("qid", "") or qid
        if isinstance(part, dict):
            qid = part.get("qid", "") or qid
        if images:
            logger.info(
                f"[Multimodal Debug] HF bonus qid={qid}: calling pipeline with "
                f"{len(leadin_images)} leadin + {len(part_images)} part = {len(images)} image(s): "
                f"{_summarize_images(images)}"
            )
        else:
            logger.info(
                f"[Multimodal Debug] HF bonus qid={qid}: calling pipeline with no images (text-only)"
            )
        out = self.pipe({"leadin": leadin_text, "part": part_text, "images": images})
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if not isinstance(out, dict):
            raise TypeError(f"HF bonus pipeline must return a dict, got {type(out)}")
        guess = out.get("answer")
        guess = "" if guess is None else str(guess)
        confidence = float(out.get("confidence", 0.0))
        expl = out.get("explanation")
        explanation = "" if expl is None else str(expl)
        return {
            "guess": guess,
            "confidence": confidence,
            "logprob": None,
            "explanation": explanation,
            "step_contents": [],
            "response_time": 0.0,
            "step_outputs": {},
        }


def _full_hf_model_id(submission: dict) -> str:
    """Hub id ``{username}/{model_name}``, or a local directory path.

    A submission may carry ``local_path`` (set by generate_outputs.py for
    locally built pipeline repos, e.g. qanta-submit/build/<short>); local
    directories are passed straight to ``transformers.pipeline`` so small
    open VLMs can be evaluated without pushing to the Hub.
    """
    local_path = submission.get("local_path")
    if local_path:
        return str(local_path)
    return f"{submission['username']}/{submission['model_name']}"


def _hf_pipeline_model_kwargs() -> dict[str, Any]:
    """Always use NF4 4-bit on CUDA for ``pipeline(..., model_kwargs=...)`` (e.g. T4 eval).

    Falls back to ``{}`` on CPU or if ``bitsandbytes`` is missing. Set
    ``HF_PIPELINE_DISABLE_4BIT=1`` to force full-precision on GPU (debug / rare models).
    """
    import torch

    def _full_precision_gpu_kwargs() -> dict[str, Any]:
        """bf16 + device_map=auto on CUDA (previously fell back to CPU-{})."""
        if not torch.cuda.is_available():
            return {}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"HF pipeline: loading full-precision {dtype} with device_map=auto on CUDA.")
        return {"torch_dtype": dtype, "device_map": "auto"}

    if os.environ.get("HF_PIPELINE_DISABLE_4BIT", "").strip().lower() in ("1", "true", "yes", "on"):
        logger.info("HF pipeline: HF_PIPELINE_DISABLE_4BIT set; loading without 4-bit quantization.")
        return _full_precision_gpu_kwargs()

    if not torch.cuda.is_available():
        logger.info("HF pipeline: no CUDA device; loading model without 4-bit quantization.")
        return {}

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        logger.warning(f"HF pipeline: 4-bit skipped (install bitsandbytes on GPU images): {e}")
        return _full_precision_gpu_kwargs()

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    logger.info("HF pipeline: loading model in 4-bit (NF4) on CUDA.")
    return {"quantization_config": qconf, "device_map": "auto"}


def _strip_cross_repo_auto_map(model_id: str, cfg: dict) -> dict:
    """Drop ``auto_map`` when it references another Hub repo than ``model_id``.

    Fine-tunes often copy a base ``config.json`` whose ``auto_map`` points at the
    base org (e.g. ``Qwen/Qwen2-VL-2B-Instruct--configuration_qwen2_vl...``). Those
    remote Python files were removed from the base repo after native transformers
    support; keeping ``auto_map`` breaks ``trust_remote_code=True`` with OSError.
    Removing it lets ``model_type`` / ``architectures`` resolve via built-in classes.
    """
    am = cfg.get("auto_map")
    if not isinstance(am, dict):
        return cfg
    for v in am.values():
        if not isinstance(v, str) or "--" not in v:
            continue
        ref_repo, _, _ = v.partition("--")
        if "/" in ref_repo and ref_repo != model_id:
            out = dict(cfg)
            out.pop("auto_map", None)
            return out
    return cfg


def _auto_config_for_hf_pipeline(model_id: str):
    """Load Hub ``config.json`` and return ``AutoConfig`` safe for ``pipeline()``."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    if os.path.isdir(model_id):
        path = os.path.join(model_id, "config.json")
    else:
        path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    had_auto_map = isinstance(raw.get("auto_map"), dict)
    sanitized = _strip_cross_repo_auto_map(model_id, raw)
    if had_auto_map and "auto_map" not in sanitized:
        logger.info(
            f"HF pipeline {model_id}: removed cross-repo auto_map so config/model "
            f"classes resolve from transformers (custom_pipelines unchanged)."
        )
    # AutoConfig.from_dict is not available on all transformers versions; load from a
    # temp config.json so AutoConfig.from_pretrained resolves via model_type.
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = os.path.join(tmp, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as wf:
            json.dump(sanitized, wf)
        return AutoConfig.from_pretrained(tmp, trust_remote_code=False)


def _hf_pipeline_tokenizer(model_id: str):
    """Custom Hub pipelines often use ``self.tokenizer``; ``pipeline()`` may leave it unset."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def run_hf_pipeline_tossup(
    submission: dict, dataset: Dataset, *, early_stop: bool = True
) -> list[dict]:
    model_id = _full_hf_model_id(submission)
    logger.info(f"Loading HF pipeline task=quizbowl-tossup model={model_id}")
    _ensure_pipeline_registry_compat()
    _patch_cached_hub_pipeline(model_id)
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "quizbowl-tossup",
        model=model_id,
        tokenizer=_hf_pipeline_tokenizer(model_id),
        trust_remote_code=True,
        config=_auto_config_for_hf_pipeline(model_id),
        model_kwargs=_hf_pipeline_model_kwargs(),
    )
    agent = _HfTossupPipelineAgent(pipe)
    return run_and_eval_tossup_dataset(
        agent, dataset, num_workers=1, early_stop=early_stop
    )


def run_hf_pipeline_bonus(submission: dict, dataset: Dataset) -> list[dict]:
    model_id = _full_hf_model_id(submission)
    logger.info(f"Loading HF pipeline task=quizbowl-bonus model={model_id}")
    _ensure_pipeline_registry_compat()
    _patch_cached_hub_pipeline(model_id)
    from transformers import pipeline as hf_pipeline

    pipe = hf_pipeline(
        "quizbowl-bonus",
        model=model_id,
        tokenizer=_hf_pipeline_tokenizer(model_id),
        trust_remote_code=True,
        config=_auto_config_for_hf_pipeline(model_id),
        model_kwargs=_hf_pipeline_model_kwargs(),
    )
    agent = _HfBonusPipelineAgent(pipe)
    return run_and_eval_bonus_dataset(agent, dataset, num_workers=1)
