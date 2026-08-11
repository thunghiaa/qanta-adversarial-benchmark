"""
Configuration settings for the workflows package.

This module contains configuration settings and constants used across the workflows package,
including model configurations, workflow settings, and other package-wide constants.
"""

AVAILABLE_MODELS = {
    "DeepSeek/V3": {
        "model": "deepseek-chat",
        "logprobs": False,
        "cost_per_million": 0.27,
    },
    "OpenAI/gpt-4.1": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 2.0,
    },
    "OpenAI/gpt-4.1-mini": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 0.4,
    },
    "OpenAI/gpt-4.1-nano": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 0.1,
    },
    "OpenAI/gpt-4o": {
        "model": "gpt-4o-2024-11-20",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "OpenAI/gpt-4o-mini": {
        "model": "gpt-4o-mini-2024-07-18",
        "logprobs": True,
        "cost_per_million": 0.15,
    },
    "OpenAI/gpt-3.5-turbo": {
        "model": "gpt-3.5-turbo-0125",
        "cost_per_million": 0.15,
    },
    "Anthropic/claude-3-7-sonnet": {
        "model": "claude-3-7-sonnet-20250219",
        "cost_per_million": 3.0,
    },
    "Anthropic/claude-sonnet-4-6": {
        "model": "claude-sonnet-4-6",
        "cost_per_million": 3.0,
    },
    "Anthropic/claude-3-5-sonnet": {
        "model": "claude-3-5-sonnet-20241022",
        "cost_per_million": 3.0,
    },
    "Anthropic/claude-3-5-haiku": {
        "model": "claude-3-5-haiku-20241022",
        "cost_per_million": 0.80,
    },
    "Cohere/command-a": {
        "model": "command-a-03-2025",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "Cohere/command-r-plus": {
        "model": "command-r-plus-08-2024",
        "logprobs": True,
        "cost_per_million": 2.50,
    },
    "Cohere/command-r": {
        "model": "command-r-08-2024",
        "logprobs": True,
        "cost_per_million": 0.15,
    },
    "Cohere/command-r7b": {
        "model": "command-r7b-12-2024",
        "logprobs": False,
        "cost_per_million": 0.0375,
    },
    "Together/Qwen3-8B": {
        "model": "Qwen/Qwen3-8B",
        "logprobs": False,
        "cost_per_million": 0.18,
    },
    # ------------------------------------------------------------------
    # 2026-07 additions for the backbone-sweep experiments (add-only; do
    # NOT modify the entries above — they document what past runs called).
    # NOTE: the legacy "OpenAI/gpt-4.1*" entries above all resolve to
    # gpt-4o-2024-11-20; "gpt-4.1-mini-true" below is the real 4.1-mini,
    # added to audit that discrepancy empirically.
    # ------------------------------------------------------------------
    "OpenAI/gpt-4.1-mini-true": {
        "model": "gpt-4.1-mini-2025-04-14",
        "logprobs": True,
        "cost_per_million": 0.4,
    },
    "OpenAI/gpt-5-mini": {
        "model": "gpt-5-mini",
        "logprobs": False,  # reasoning family: no logprobs
        "cost_per_million": 0.25,
    },
    "OpenAI/gpt-5.1": {
        "model": "gpt-5.1",
        "logprobs": False,
        "cost_per_million": 1.25,
    },
    "Anthropic/claude-haiku-4-5": {
        "model": "claude-haiku-4-5-20251001",
        "cost_per_million": 1.0,
    },
    "Anthropic/claude-sonnet-5": {
        "model": "claude-sonnet-5",
        "cost_per_million": 3.0,
    },
    "Anthropic/claude-opus-4-8": {
        "model": "claude-opus-4-8",
        "cost_per_million": 5.0,
    },
    # 2026-08 additions: GPT-5.6 family (Jordan's ask, meeting 2026-07-31 —
    # add modern models to the analysis). Verified live on us.api.openai.com.
    "OpenAI/gpt-5.6-sol": {
        "model": "gpt-5.6-sol",
        "logprobs": False,  # reasoning family: no logprobs
        "cost_per_million": 5.0,
    },
    "OpenAI/gpt-5.6-terra": {
        "model": "gpt-5.6-terra",
        "logprobs": False,
        "cost_per_million": 2.5,
    },
    "OpenAI/gpt-5.6-luna": {
        "model": "gpt-5.6-luna",
        "logprobs": False,
        "cost_per_million": 1.0,
    },
}

# Function mapping for input/output transformations
TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}

FUNCTION_MAP = {
    "upper": str.upper,
    "lower": str.lower,
    "len": len,
    "split": str.split,
}
