import itertools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.envs import API, LOCAL_LOGS_DIR, LOG_LEVEL, Q25_CACHE_REPO

os.makedirs(LOCAL_LOGS_DIR, exist_ok=True)
log_file = Path(LOCAL_LOGS_DIR) / "output.log"


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def _loguru_filter(record, every_n_counters={}):
    msg = record["message"]
    tag = record["extra"].get("every_n")  # only messages with this tag are sampled
    if tag is None:  # un-tagged → always log
        return True
    if isinstance(freq := tag, int):
        ctr = every_n_counters.setdefault(msg, itertools.count())
        return next(ctr) % freq == 0  # True only on 0, n, 2n, …
    else:
        return True


loguru_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<magenta>P:{process} T:{thread}</magenta> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def push_logs_to_dataset():
    repo_filename = str(log_file.resolve()).replace(".log", f"-{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log")
    repo_filepath = f"logs/{repo_filename}"
    API.upload_file(path_or_fileobj=log_file, path_in_repo=repo_filepath, repo_id=Q25_CACHE_REPO, repo_type="dataset")


def configure_root_logger():
    """
    Configure the root logger to write to a file and to the console.

    Example to use 'every_n' logging:
    logger.bind(every_n=10).info("Hello but only every 10th hot-loop")
    """
    # Configure the root logger
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()

    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # translate the loguru format string to logging format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | P:%(process)d T:%(thread)d | %(name)s:%(lineno)d | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    from loguru import logger

    logger.remove()
    logger.add(sys.stdout, level=LOG_LEVEL, diagnose=False, colorize=True, filter=_loguru_filter, format=loguru_format)
    logger.add(
        log_file,
        level=LOG_LEVEL,
        diagnose=False,
        format=loguru_format,
        encoding="utf-8",
        enqueue=True,
        rotation="10 MB",
        retention="7 days",
        filter=_loguru_filter,
    )


# %%
from pathlib import Path

str(Path(".langchain.db").resolve())
# %%
