import copy
import os
import os.path as osp
import sys
from functools import reduce
from datetime import datetime, timezone

_HAS_FCNTL = sys.platform != "win32"
if _HAS_FCNTL:
    import fcntl

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from loguru import logger

from shared.workflows.metrics import (
    compare_tossup_outputs,
    compute_bonus_metrics,
    compute_tossup_metrics,
    compute_workflow_cost,
)
from shared.workflows.qb_agents import QuizBowlBonusAgent, QuizBowlTossupAgent
from shared.workflows.runners import inject_bonus_metrics, run_and_eval_bonus_dataset, run_and_eval_tossup_dataset
from shared.workflows.structs import TossupWorkflow, Workflow
from src.envs import (
    API,
    APPEND_MULTIMODAL_EVAL_TAIL,
    COMBINED_EVAL_SPLIT_NAME,
    ELO_COMPUTE_REQUEST_DIR,
    EVAL_COMPONENT_SPLITS,
    EVAL_DATASET,
    JOBS_REPO,
    LOCAL_JOBS_PATH,
    LOCAL_OUTPUT_PATH,
    LOCAL_REQUESTS_PATH,
    LOCAL_RESULTS_PATH,
    MULTIMODAL_EVAL_HUB,
    MULTIMODAL_EVAL_POOL_SIZE,
    MULTIMODAL_EVAL_ROW_SELECTION,
    MULTIMODAL_EVAL_SPLIT_NAME,
    PLAYGROUND_MAX_QUESTIONS,
    REQUESTS_REPO,
)

from .io_utils import (
    load_model_outputs,
    load_model_results,
    model_outputs_path,
    model_results_path,
    write_model_outputs,
    write_model_result,
    write_model_tossup_elo_results,
)
from .jobs import Job, JobManager, JobStage, JobStatus
from .docker_runner import run_docker_bonus, run_docker_tossup
from .hf_pipeline_runner import run_hf_pipeline_bonus, run_hf_pipeline_tossup
from .submissions import SubmissionManager

submissions_manager = SubmissionManager(LOCAL_REQUESTS_PATH, REQUESTS_REPO, API)
job_manager = JobManager(LOCAL_JOBS_PATH, JOBS_REPO, submissions_manager, API)


def compute_submission_cost(submission_id: str) -> float:
    data = submissions_manager.get_submission(submission_id)
    if data is None or not data.get("workflow"):
        return 0.0
    workflow = Workflow(**data["workflow"])
    return compute_workflow_cost(workflow)["cost"]


def generate_tossup_outputs(submission: dict, dataset: Dataset) -> list[dict]:
    if submission.get("submission_type") == "docker_image":
        return run_docker_tossup(submission, dataset)
    if submission.get("submission_type") == "hf_pipeline":
        return run_hf_pipeline_tossup(submission, dataset)
    tossup_agent = QuizBowlTossupAgent(workflow=TossupWorkflow(**submission["workflow"]))
    return run_and_eval_tossup_dataset(tossup_agent, dataset)


def generate_bonus_outputs(submission: dict, dataset: Dataset) -> list[dict]:
    if submission.get("submission_type") == "docker_image":
        return run_docker_bonus(submission, dataset)
    if submission.get("submission_type") == "hf_pipeline":
        return run_hf_pipeline_bonus(submission, dataset)
    bonus_agent = QuizBowlBonusAgent(workflow=Workflow(**submission["workflow"]))
    return run_and_eval_bonus_dataset(bonus_agent, dataset)


def create_result_dict(job: Job, metrics: dict, n_examples: int, **kwargs) -> dict:
    sub = submissions_manager.get_submission(job.submission_id)
    submission_created_at = sub.get("created_at") if sub else None
    out = {
        "id": job.submission_id,
        "competition_type": job.competition_type,
        "username": job.username,
        "model_name": job.model_name,
        "eval_set": job.eval_split,
        "metrics": metrics,
        "n_examples": n_examples,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    if submission_created_at:
        out["submission_created_at"] = submission_created_at
    return out


def _drop_multimodal_rows_requiring_audio_decode(ds: Dataset) -> Dataset:
    """
    Remove rows whose leadin_audio / part_audio lists are non-empty (would require torchcodec on decode).
    Uses PyArrow only so we never materialize Audio through HF's Python formatter.
    """
    audio_cols = [c for c in ("leadin_audio", "part_audio") if c in ds.column_names]
    if not audio_cols:
        return ds
    table = ds.data.table
    keep: list[int] = []
    for i in range(table.num_rows):
        needs_audio = False
        for c in audio_cols:
            scalar = table.column(c)[i]
            if scalar.is_valid and len(scalar) > 0:
                needs_audio = True
                break
        if not needs_audio:
            keep.append(i)
    dropped = len(ds) - len(keep)
    if dropped:
        logger.info(
            f"Excluded {dropped} multimodal row(s) that require audio decoding "
            f"(kept {len(keep)}/{len(ds)})."
        )
    if not keep:
        return ds.select([])
    return ds.select(keep)


def _strip_multimodal_audio_columns(ds: Dataset) -> Dataset:
    """Drop Audio-typed columns so row access never triggers audio decoding (e.g. torchcodec)."""
    to_drop = [c for c in ("leadin_audio", "part_audio") if c in ds.column_names]
    return ds.remove_columns(to_drop) if to_drop else ds


def _load_raw_multimodal_train(competition_type: str) -> Dataset:
    """Load train split from MULTIMODAL_EVAL_HUB; subset bonus vs tossups matches competition_type."""
    subset = "bonus" if competition_type == "bonus" else "tossups"
    ds = load_dataset(MULTIMODAL_EVAL_HUB, subset, split="train", trust_remote_code=False)
    ds = _drop_multimodal_rows_requiring_audio_decode(ds)
    ds = _strip_multimodal_audio_columns(ds)
    return ds


def _multimodal_tail_window_from_train(raw: Dataset) -> Dataset:
    """Rows [PLAYGROUND_MAX_QUESTIONS : min(len, MULTIMODAL_EVAL_POOL_SIZE)) (same ordering as the playground)."""
    pool_cap = int(MULTIMODAL_EVAL_POOL_SIZE)
    start = int(PLAYGROUND_MAX_QUESTIONS)
    n_pool = min(pool_cap, len(raw))
    pool = raw.select(range(n_pool))
    if start >= len(pool):
        return pool.select([])
    return pool.select(range(start, len(pool)))


def _load_multimodal_eval_tail(competition_type: str) -> Dataset:
    """Legacy tail append: same rows as dedicated multimodal split when ROW_SELECTION=tail_window."""
    raw = _load_raw_multimodal_train(competition_type)
    return _multimodal_tail_window_from_train(raw)


def _load_multimodal_eval_split_dataset(competition_type: str) -> Dataset:
    """Eval split MULTIMODAL_EVAL_SPLIT_NAME: load from Hub only (no qanta25-eval)."""
    raw = _load_raw_multimodal_train(competition_type)
    if MULTIMODAL_EVAL_ROW_SELECTION == "full_train":
        logger.info(
            f"Multimodal eval split: full train from {MULTIMODAL_EVAL_HUB} "
            f"({'bonus' if competition_type == 'bonus' else 'tossups'}), n={len(raw)}"
        )
        return raw
    ds = _multimodal_tail_window_from_train(raw)
    logger.info(
        f"Multimodal eval split: tail_window from {MULTIMODAL_EVAL_HUB} "
        f"({'bonus' if competition_type == 'bonus' else 'tossups'}), n={len(ds)}"
    )
    return ds


# qanta25-eval bonus `parts` vs multimodal Hub bonus `parts` use different Arrow struct fields; align before concat.
_BONUS_PART_UNION_KEYS = (
    "answer_line",
    "answer_primary",
    "clean_answers",
    "difficulty_modifier",
    "explanation",
    "number",
    "question",
    "value",
    "answer",
    "answer_img",
    "answer_refs",
    "image_relative_paths",
    "multimodal_tokens",
)

# If we use None for missing strings, Arrow infers Value("null") for one chunk and Value("string") for another → concat fails.
_BONUS_PART_STRING_KEYS = frozenset(
    {
        "answer_line",
        "answer_primary",
        "difficulty_modifier",
        "explanation",
        "answer",
        "answer_img",
    }
)

# List-of-struct for `parts` in Features uses an inner dict wrapped in a list (see HF Features docs).
_UNIFIED_BONUS_PART_ELEM = {
    "answer_line": Value("string"),
    "answer_primary": Value("string"),
    "clean_answers": Sequence(Value("string")),
    "difficulty_modifier": Value("string"),
    "explanation": Value("string"),
    "number": Value("int64"),
    "question": Value("string"),
    "value": Value("int64"),
    "answer": Value("string"),
    "answer_img": Value("string"),
    "answer_refs": Sequence(Value("string")),
    "image_relative_paths": Sequence(Value("string")),
    "multimodal_tokens": [
        {
            "content": Value("string"),
            "hash_key": Value("string"),
            "position": Value("int64"),
            "type": Value("string"),
        }
    ],
}


def _normalize_multimodal_tokens(val: object) -> list[dict]:
    """Stable struct for every token so List({...}) schema matches across chunks."""
    if not val:
        return []
    out: list[dict] = []
    for t in val:
        if not isinstance(t, dict):
            continue
        pos = t.get("position")
        out.append(
            {
                "content": t.get("content") if t.get("content") is not None else "",
                "hash_key": t.get("hash_key") if t.get("hash_key") is not None else "",
                "position": int(pos) if pos is not None else 0,
                "type": t.get("type") if t.get("type") is not None else "",
            }
        )
    return out


def _normalize_bonus_part_dict(p: object) -> dict:
    """Single bonus part row with a fixed key set so merged datasets share one `parts` schema."""
    if not isinstance(p, dict):
        p = {}
    out: dict = {}
    for key in _BONUS_PART_UNION_KEYS:
        val = p.get(key)
        if key == "clean_answers":
            if val is None:
                out[key] = []
            else:
                out[key] = [str(x) for x in val if x is not None]
        elif key == "multimodal_tokens":
            out[key] = _normalize_multimodal_tokens(val)
        elif key == "image_relative_paths":
            if val is None:
                out[key] = []
            else:
                out[key] = [str(x) for x in val if x is not None]
        elif key == "answer_refs":
            if val is None:
                out[key] = []
            else:
                out[key] = [str(x) for x in val if x is not None]
        elif key == "number":
            v = p.get("number")
            out[key] = int(v) if v is not None else 0
        elif key == "question":
            q = p.get("question")
            if q is None:
                q = p.get("part")
            out[key] = str(q) if q is not None else ""
        elif key == "value":
            v = p.get("value")
            out[key] = int(v) if v is not None else 0
        elif key in _BONUS_PART_STRING_KEYS:
            out[key] = "" if val is None else str(val)
        else:
            raise RuntimeError(f"Unhandled bonus part key {key!r}; update _normalize_bonus_part_dict")
    return out


def _normalize_bonus_example_for_merge(example: dict) -> dict:
    row = dict(example)
    parts = row.get("parts")
    if parts is None:
        row["parts"] = []
    else:
        row["parts"] = [_normalize_bonus_part_dict(p) for p in parts]
    return row


def _dataset_with_unified_bonus_parts(ds: Dataset) -> Dataset:
    """Rebuild bonus dataset so `parts` uses one struct schema (needed to stack qanta25-eval + multimodal Hub)."""
    if len(ds) == 0:
        return ds
    rows = [_normalize_bonus_example_for_merge(ds[i]) for i in range(len(ds))]
    feat = {name: ds.features[name] for name in ds.column_names}
    feat["parts"] = [_UNIFIED_BONUS_PART_ELEM]
    return Dataset.from_list(rows, features=Features(feat))


_TOSSUP_LIST_STRING_COLS = ("clean_answers", "answer_refs", "image_relative_paths")


def _dataset_with_unified_tossup_list_features(ds: Dataset) -> Dataset:
    """Normalize tossup list columns to Sequence(...) so qanta25-eval + multimodal Hub can concat."""
    if len(ds) == 0:
        return ds
    cols = [c for c in _TOSSUP_LIST_STRING_COLS if c in ds.column_names]
    if not cols and "run_indices" not in ds.column_names:
        return ds
    rows: list[dict] = []
    for i in range(len(ds)):
        row = dict(ds[i])
        for col in cols:
            val = row.get(col)
            row[col] = [] if val is None else [str(x) for x in val if x is not None]
        if "run_indices" in row:
            val = row.get("run_indices")
            row["run_indices"] = [] if val is None else [int(x) for x in val]
        rows.append(row)
    feat = {name: ds.features[name] for name in ds.column_names}
    for col in cols:
        feat[col] = Sequence(Value("string"))
    if "run_indices" in feat:
        feat["run_indices"] = Sequence(Value("int64"))
    return Dataset.from_list(rows, features=Features(feat))


def _merge_two_eval_datasets(base: Dataset, tail: Dataset) -> Dataset:
    """
    Vertically stack two eval datasets with aligned schemas (e.g. qanta25-eval + multimodal Hub).

    Either side may lack `metadata` or multimodal columns; we pad with nulls / default metadata.
    """
    tail_w = tail
    base_w = base

    if "metadata" in base_w.column_names and "metadata" not in tail_w.column_names:
        meta_feat = base_w.features["metadata"]
        if len(base_w) > 0:
            meta_col = []
            for _ in range(len(tail_w)):
                m = copy.deepcopy(base_w[0]["metadata"])
                if isinstance(m, dict):
                    m["human_buzz_positions"] = None
                meta_col.append(m)
        else:
            meta_col = [{"human_buzz_positions": None} for _ in range(len(tail_w))]
        tail_w = tail_w.add_column("metadata", meta_col, feature=meta_feat)

    for col in base_w.column_names:
        if col not in tail_w.column_names:
            tail_w = tail_w.add_column(
                col,
                [None] * len(tail_w),
                feature=base_w.features[col],
            )

    for col in list(tail_w.column_names):
        if col not in base_w.column_names:
            base_w = base_w.add_column(
                col,
                [None] * len(base_w),
                feature=tail_w.features[col],
            )

    all_cols = list(dict.fromkeys(list(base.column_names) + list(tail.column_names)))
    base_w = base_w.select_columns(all_cols)
    tail_w = tail_w.select_columns(all_cols)
    if (
        "parts" in base_w.column_names
        and "parts" in tail_w.column_names
        and base_w.features["parts"] != tail_w.features["parts"]
    ):
        base_w = _dataset_with_unified_bonus_parts(base_w)
        tail_w = _dataset_with_unified_bonus_parts(tail_w)
    if "clean_answers" in base_w.column_names or "clean_answers" in tail_w.column_names:
        if base_w.features.get("clean_answers") != tail_w.features.get("clean_answers"):
            base_w = _dataset_with_unified_tossup_list_features(base_w)
            tail_w = _dataset_with_unified_tossup_list_features(tail_w)
    return concatenate_datasets([base_w, tail_w])


def _merge_eval_with_multimodal_tail(base: Dataset, tail: Dataset) -> Dataset:
    """Backward-compatible name: append multimodal tail to a qanta25-eval split."""
    return _merge_two_eval_datasets(base, tail)


def _load_single_eval_component(competition_type: str, split: str) -> Dataset:
    """Load one named split for building the combined suite (no recursive combined_eval, no tail append)."""
    es = split.strip()
    if MULTIMODAL_EVAL_SPLIT_NAME and es == MULTIMODAL_EVAL_SPLIT_NAME.strip():
        if competition_type not in ("tossup", "bonus"):
            raise ValueError(f"Multimodal eval requires competition_type tossup or bonus, got {competition_type}")
        return _load_multimodal_eval_split_dataset(competition_type)
    return load_dataset(EVAL_DATASET, competition_type, split=es)


def _load_combined_eval_dataset(competition_type: str) -> Dataset:
    """Concatenate EVAL_COMPONENT_SPLITS into one evaluation pass (metrics are global over all rows)."""
    if competition_type not in ("tossup", "bonus"):
        raise ValueError(f"Combined eval requires competition_type tossup or bonus, got {competition_type}")
    if not EVAL_COMPONENT_SPLITS:
        raise ValueError("EVAL_COMPONENT_SPLITS is empty; cannot build combined eval dataset.")
    parts = [_load_single_eval_component(competition_type, s) for s in EVAL_COMPONENT_SPLITS]
    merged = reduce(_merge_two_eval_datasets, parts)
    logger.info(
        f"Combined eval dataset ({COMBINED_EVAL_SPLIT_NAME}): components={EVAL_COMPONENT_SPLITS}, n={len(merged)}"
    )
    return merged


def load_eval_dataset(job: Job) -> Dataset:
    es = job.eval_split.strip()
    if COMBINED_EVAL_SPLIT_NAME and es == COMBINED_EVAL_SPLIT_NAME.strip():
        if job.competition_type not in ("tossup", "bonus"):
            raise ValueError(f"Combined eval split requires competition_type tossup or bonus, got {job.competition_type}")
        return _load_combined_eval_dataset(job.competition_type)

    # Dedicated eval split: only multimodal Hub data (subset tossups vs bonus from competition_type).
    if ":" not in job.eval_split and MULTIMODAL_EVAL_SPLIT_NAME and es == MULTIMODAL_EVAL_SPLIT_NAME.strip():
        if job.competition_type not in ("tossup", "bonus"):
            raise ValueError(f"Multimodal eval split requires competition_type tossup or bonus, got {job.competition_type}")
        try:
            return _load_multimodal_eval_split_dataset(job.competition_type)
        except Exception as e:
            logger.warning(f"Could not load multimodal eval split ({e}).")
            raise

    if ":" in job.eval_split:
        ds_name, split = job.eval_split.split(":")
    else:
        ds_name = EVAL_DATASET
        split = job.eval_split
    dataset = load_dataset(ds_name, job.competition_type, split=split)

    if (
        not APPEND_MULTIMODAL_EVAL_TAIL
        or job.competition_type not in ("tossup", "bonus")
        or ":" in job.eval_split
    ):
        return dataset

    try:
        tail = _load_multimodal_eval_tail(job.competition_type)
        if len(tail) == 0:
            return dataset
        overlap = [c for c in dataset.column_names if c in tail.column_names]
        if not overlap:
            logger.warning("Multimodal eval tail skipped: no overlapping columns with base eval dataset.")
            return dataset
        merged = _merge_eval_with_multimodal_tail(dataset, tail)
        logger.info(
            f"Eval dataset merged: base n={len(dataset)}, multimodal tail n={len(tail)}, combined n={len(merged)}"
        )
        return merged
    except Exception as e:
        logger.warning(f"Could not append multimodal eval tail ({e}); using base eval split only.")
        return dataset


def run_bonus_outputs_gen(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH):
    logger.info(f"Starting bonus outputs generation for job {job.id}")
    job = job_manager.update_job_status(job, JobStatus.IN_PROGRESS)
    dataset = load_eval_dataset(job)
    system_outputs = generate_bonus_outputs(job_manager.get_submission(job), dataset)
    write_model_outputs(system_outputs, job, outputs_repo_dir)
    return job_manager.update_job_stage(job, JobStage.METRICS)


def run_bonus_metrics_gen(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH):
    logger.info(f"Starting bonus metrics generation for job {job.id}")
    job = job_manager.update_job_status(job, JobStatus.IN_PROGRESS)
    logger.info(f"Job {job.id}:{job.stage} changing status from {job.status} to {JobStatus.IN_PROGRESS}")
    dataset = load_eval_dataset(job)
    system_outputs = load_model_outputs(job, outputs_repo_dir)
    system_outputs = inject_bonus_metrics(system_outputs, dataset)
    write_model_outputs(system_outputs, job, outputs_repo_dir)
    return job_manager.update_job_stage(job, JobStage.EVALUATION)


def run_bonus_eval(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH, results_repo_dir: str = LOCAL_RESULTS_PATH):
    logger.info(f"Starting bonus evaluation for job {job.id}")
    job = job_manager.update_job_status(job, JobStatus.IN_PROGRESS)
    system_outputs = load_model_outputs(job, outputs_repo_dir)
    metrics = compute_bonus_metrics(system_outputs)
    n_parts = sum(len(out["part_outputs"]) for out in system_outputs)
    model_result = create_result_dict(job, metrics, n_examples=len(system_outputs), n_parts=n_parts)
    write_model_result(model_result, job, results_repo_dir)
    return job_manager.update_job_status(job, JobStatus.COMPLETED)


def run_tossup_outputs_gen(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH):
    logger.info(f"Starting tossup outputs generation for job {job.id}")
    logger.info(f"Job {job.id}:{job.stage} changing status from {job.status} to {JobStatus.IN_PROGRESS}")
    job = job_manager.update_job_status(job, JobStatus.IN_PROGRESS)
    output_filepath = model_outputs_path(job, outputs_repo_dir)
    if not osp.exists(output_filepath):
        dataset = load_eval_dataset(job)
        system_outputs = generate_tossup_outputs(job_manager.get_submission(job), dataset)
        write_model_outputs(system_outputs, job, outputs_repo_dir)
    else:
        logger.info(f"Job {job.id} already generated outputs. Skipping for generating model outputs.")
    request_elo_computation(job.eval_split)
    return job_manager.update_job_stage(job, JobStage.EVALUATION)


def run_tossup_eval(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH, results_repo_dir: str = LOCAL_RESULTS_PATH):
    logger.info(f"Starting tossup evaluation for job {job.id}")
    job = job_manager.update_job_status(job, JobStatus.IN_PROGRESS)
    system_outputs = load_model_outputs(job, outputs_repo_dir)
    dataset = load_eval_dataset(job)

    # Compute the cost of the workflow
    cost = compute_submission_cost(job.submission_id)

    # Compute the metrics
    human_buzz_positions = []
    for md in dataset["metadata"]:
        h = md.get("human_buzz_positions") if isinstance(md, dict) else None
        if h is not None and len(h) == 0:
            h = None
        human_buzz_positions.append(h)
    metrics = compute_tossup_metrics(system_outputs, dataset["run_indices"], human_buzz_positions)

    # Create the result dict and write it to the results repo
    model_result = create_result_dict(job, metrics, n_examples=len(dataset), cost=cost)
    write_model_result(model_result, job, results_repo_dir)

    # Check for the elo computation request and mark it as completed if it exists
    if check_elo_compute_request(job.eval_split):
        mark_elo_compute_request_completed(job.eval_split)
    return job_manager.update_job_stage(job, JobStage.EVALUATION, JobStatus.COMPLETED)


def run_bonus_job(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH, results_repo_dir: str = LOCAL_RESULTS_PATH):
    if job.status in {JobStatus.COMPLETED, JobStatus.IN_PROGRESS}:
        logger.info(f"Job {job.id} already {job.status}. This shouldn't happen. Check your code for race conditions.")
        return

    if job.stage == JobStage.OUTPUTS:
        return run_bonus_outputs_gen(job, outputs_repo_dir)

    elif job.stage == JobStage.METRICS:
        return run_bonus_metrics_gen(job, outputs_repo_dir)

    elif job.stage == JobStage.EVALUATION:
        return run_bonus_eval(job, outputs_repo_dir, results_repo_dir)


def run_tossup_job(job: Job, outputs_repo_dir: str = LOCAL_OUTPUT_PATH, results_repo_dir: str = LOCAL_RESULTS_PATH):
    if job.status in {JobStatus.IN_PROGRESS, JobStatus.COMPLETED}:
        logger.warning(
            f"Job {job.id} is still {job.status}. This shouldn't happen. Check your code for race conditions."
        )
        return

    if job.stage == JobStage.OUTPUTS:
        return run_tossup_outputs_gen(job, outputs_repo_dir)

    elif job.stage == JobStage.EVALUATION:
        return run_tossup_eval(job, outputs_repo_dir, results_repo_dir)


# -------------------------- ELO (MODEL-MODEL SIMULATION) FUNCTIONS --------------------------


def simulate_tossup_eval(system_outputs_1: list[dict], system_outputs_2: list[dict], tie_strategy: str = "first"):
    points_1, points_2 = 0, 0
    if len(system_outputs_1) != len(system_outputs_2):
        raise ValueError(f"System outputs have different lengths: {len(system_outputs_1)} != {len(system_outputs_2)}")
    for system1_output, system2_output in zip(system_outputs_1, system_outputs_2):
        if system1_output["qid"] != system2_output["qid"]:
            raise ValueError("System outputs have different qids")
        points = compare_tossup_outputs(system1_output, system2_output, tie_strategy)
        points_1 += points[0]
        points_2 += points[1]
    if points_1 > len(system_outputs_1):
        raise ValueError(
            f"Points 1 is greater than the number of system outputs 1: {points_1} > {len(system_outputs_1)}"
        )
    if points_2 > len(system_outputs_2):
        raise ValueError(
            f"Points 2 is greater than the number of system outputs 2: {points_2} > {len(system_outputs_2)}"
        )
    if points_1 + points_2 > 2 * len(system_outputs_1):
        raise ValueError(
            f"Points 1 + points 2 is greater than 2 * the number of system outputs 1: {points_1 + points_2} > {2 * len(system_outputs_1)}"
        )
    return points_1, points_2


class FileRWLock:
    """
    File-based read/write lock using fcntl on Unix.
    On Windows, opens the file without locking (no fcntl); fine for local dev.
    Use mode='r' for shared (read) lock, mode='w' for exclusive (write) lock.
    """

    def __init__(self, filename, mode="w", timeout=60):
        self.filename = filename
        self.mode = mode
        self.timeout = timeout
        self.file = None
        if _HAS_FCNTL:
            self.lock_mode = fcntl.LOCK_SH if self.mode in {"r", "rb"} else fcntl.LOCK_EX

    def __enter__(self):
        # Open file for reading or appending, depending on mode
        self.file = open(self.filename, self.mode)
        if _HAS_FCNTL:
            fcntl.flock(self.file, self.lock_mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            if _HAS_FCNTL:
                fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()


def get_elo_compute_request_file(eval_split: str):
    return f"{ELO_COMPUTE_REQUEST_DIR}/{eval_split}.txt"


def request_elo_computation(eval_split: str):
    req_file = get_elo_compute_request_file(eval_split)
    os.makedirs(osp.dirname(req_file), exist_ok=True)
    with FileRWLock(req_file, mode="w"):
        with open(req_file, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())


def mark_elo_compute_request_completed(eval_split: str):
    req_file = get_elo_compute_request_file(eval_split)
    with FileRWLock(req_file, mode="w"):
        with open(req_file, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())


def check_elo_compute_request(eval_split: str):
    req_file = get_elo_compute_request_file(eval_split)
    if not osp.exists(req_file):
        return False
    with FileRWLock(req_file, mode="r"):
        with open(req_file, "r") as f:
            return f.read().strip() != ""


def check_and_run_elo_computation(eval_split: str):
    if not check_elo_compute_request(eval_split):
        return
    mark_elo_compute_request_completed(eval_split)

    jobs = job_manager.get_jobs(competition_type="tossup", stage=JobStage.EVALUATION, eval_split=eval_split)
    if not jobs:
        logger.warning(f"Elo computation for {eval_split}: no evaluation-stage tossup jobs; skipping.")
        return

    sys_costs = {}
    sys_outputs = {}

    for job in jobs:
        if job.submission_id in sys_outputs:
            logger.error(f"Model {job.submission_id} already in sys_outputs_board")
            continue
        if (submission := submissions_manager.get_submission(job.submission_id)) is None:
            logger.error(f"Submission {job.submission_id} not found")
            continue
        sys_outputs[job.submission_id] = load_model_outputs(job, LOCAL_OUTPUT_PATH)
        workflow = Workflow(**submission["workflow"])
        sys_costs[job.submission_id] = compute_workflow_cost(workflow)["cost"]

    points_table = dict.fromkeys(sys_outputs.keys(), 0)
    # Sort the systems by cost and compute the points
    system_names = sorted(sys_outputs.keys(), key=lambda x: sys_costs[x])
    for i in range(len(system_names)):
        for j in range(i + 1, len(system_names)):
            name1, name2 = system_names[i], system_names[j]
            outputs1, outputs2 = sys_outputs[name1], sys_outputs[name2]
            points1, points2 = simulate_tossup_eval(outputs1, outputs2, tie_strategy="first")
            points_table[name1] += points1
            points_table[name2] += points2

    # Normalize the points.
    if len(system_names) > 1:
        n_matches = (len(system_names) - 1) * len(sys_outputs[system_names[0]])
        points_table = {k: v / n_matches for k, v in points_table.items()}
    write_model_tossup_elo_results(points_table, jobs[0])
