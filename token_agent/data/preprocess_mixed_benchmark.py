"""
Preprocess all active datasets into a unified mixed-benchmark parquet.

Usage:
    python -m token_agent.data.preprocess_mixed_benchmark \
        --local_dir ~/data/token_agent_mixed \
        --active_categories 0,1,2,3

Output columns per row:
    data_source      str              e.g. "openai/gsm8k", "squad", "searchR1_nq"
    task_category    int              0-4
    prompt           list[dict]       chat-format messages (system + user)
    env_kwargs       dict             env-specific kwargs (ground_truth, question, …)
    reward_model     dict | None      ground-truth payload for reward scoring
    extra_info       dict             index, split, tools_kwargs, …
"""

import argparse
import logging
import os
from typing import Callable, Dict, List

import pandas as pd

from token_agent.data.dataset_registry import (
    DATASET_REGISTRY,
    DatasetMeta,
    get_active_datasets,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset processors: each returns a list[dict] of unified rows
# ---------------------------------------------------------------------------

def _build_prompt(question: str, system: str = "") -> List[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return msgs


def process_gsm8k(meta: DatasetMeta, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(meta.hf_repo_id, "main", split=meta.split_map[split])
    rows = []
    for i, ex in enumerate(ds):
        answer = ex["answer"].split("####")[-1].strip()
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(ex["question"]),
            "env_kwargs": {"ground_truth": answer, "question": ex["question"],
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": answer},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def process_math(meta: DatasetMeta, split: str) -> List[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map[split])
    except Exception as e:
        logger.warning(
            "Could not load %s (hf_repo_id=%s): %s. Skipping this dataset.",
            meta.data_source,
            meta.hf_repo_id,
            e,
        )
        return []

    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(ex["problem"]),
            "env_kwargs": {"ground_truth": ex["solution"], "question": ex["problem"],
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": ex["solution"]},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def process_squad(meta: DatasetMeta, split: str) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset(meta.hf_repo_id, split=meta.split_map[split])
    rows = []
    for i, ex in enumerate(ds):
        answers = list(set(ex["answers"]["text"]))
        question = f"Context: {ex['context']}\n\nQuestion: {ex['question']}"
        rows.append({
            "data_source": meta.data_source,
            "task_category": meta.task_category,
            "prompt": _build_prompt(question),
            "env_kwargs": {"ground_truth": answers, "question": question,
                           "data_source": meta.data_source, "task_category": meta.task_category},
            "reward_model": {"ground_truth": {"target": answers}},
            "extra_info": {"index": i, "split": split},
        })
    return rows


def process_simpleqa(meta: DatasetMeta, split: str) -> List[dict]:
    """OpenAI SimpleQA – attempt HF load; fall back to placeholder."""
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("problem", ex.get("question", ""))
            answer = ex.get("answer", ex.get("solution", ""))
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": answer, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": {"target": [answer] if isinstance(answer, str) else answer}},
                "extra_info": {"index": i, "split": split},
            })
    except Exception as e:
        logger.warning("Could not load %s: %s. Creating placeholder.", meta.data_source, e)
    return rows


def process_aa_omniscience(meta: DatasetMeta, split: str) -> List[dict]:
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": answer, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": {"target": [answer] if isinstance(answer, str) else answer}},
                "extra_info": {"index": i, "split": split},
            })
    except Exception as e:
        logger.warning("Could not load %s: %s. Creating placeholder.", meta.data_source, e)
    return rows


def process_search_r1(meta: DatasetMeta, split: str) -> List[dict]:
    """
    Search-R1 datasets are already preprocessed via the existing
    verl-agent pipeline. This function downloads the raw HF data and
    converts it into the unified schema.
    """
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset(meta.hf_repo_id, split=meta.split_map.get(split, split))
        for i, ex in enumerate(ds):
            question = ex.get("question", "")
            golden = ex.get("golden_answers", ex.get("answer", []))
            if isinstance(golden, str):
                golden = [golden]
            gt = {"target": golden}
            tools_kwargs = {
                "search": {
                    "create_kwargs": {
                        "ground_truth": gt,
                        "question": question,
                        "data_source": meta.data_source,
                    }
                }
            }
            rows.append({
                "data_source": meta.data_source,
                "task_category": meta.task_category,
                "prompt": _build_prompt(question),
                "env_kwargs": {"ground_truth": gt, "question": question,
                               "data_source": meta.data_source, "task_category": meta.task_category},
                "reward_model": {"ground_truth": gt},
                "extra_info": {
                    "index": i,
                    "split": split,
                    "need_tools_kwargs": True,
                    "tools_kwargs": tools_kwargs,
                },
            })
    except Exception as e:
        logger.warning("Could not load %s: %s", meta.data_source, e)
    return rows


def process_env_placeholder(meta: DatasetMeta, split: str) -> List[dict]:
    """
    For environment-based datasets (ALFWorld, WebShop, Sokoban, …) the
    actual data lives inside the env package. We produce a lightweight
    parquet row that carries the data_source and task_category so the
    MixedEnvironmentManager can dispatch correctly at runtime.
    """
    return [{
        "data_source": meta.data_source,
        "task_category": meta.task_category,
        "prompt": _build_prompt(f"[Environment task: {meta.data_source}]"),
        "env_kwargs": {"data_source": meta.data_source, "task_category": meta.task_category},
        "reward_model": None,
        "extra_info": {"index": 0, "split": split},
    }]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_PROCESSORS: Dict[str, Callable] = {
    "openai/gsm8k": process_gsm8k,
    "lighteval/MATH": process_math,
    "squad": process_squad,
    "simpleqa": process_simpleqa,
    "aa_omniscience": process_aa_omniscience,
    "alfworld": process_env_placeholder,
    "webshop": process_env_placeholder,
    "sokoban": process_env_placeholder,
    "ezpoints": process_env_placeholder,
}

for _ds in ["searchR1_nq", "searchR1_triviaqa", "searchR1_popqa",
            "searchR1_hotpotqa", "searchR1_2wikimultihopqa",
            "searchR1_musique", "searchR1_bamboogle"]:
    _PROCESSORS[_ds] = process_search_r1

for _mm in ["mmstar", "sqa", "mmvet", "pope", "mmb", "math_v", "ai2d"]:
    _PROCESSORS[_mm] = process_env_placeholder


def process_dataset(meta: DatasetMeta, split: str) -> List[dict]:
    processor = _PROCESSORS.get(meta.data_source)
    if processor is None:
        logger.warning("No processor for %s – skipping", meta.data_source)
        return []
    return processor(meta, split)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Token-Agent mixed benchmark parquets.")
    parser.add_argument("--local_dir", default="~/data/token_agent_mixed",
                        help="Directory to save parquet files")
    parser.add_argument("--active_categories", default="0,1,2,3",
                        help="Comma-separated category IDs to include (0-4)")
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Cap rows per dataset (for quick testing)")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    active_cats = set(int(c) for c in args.active_categories.split(","))

    for split in ["train", "test"]:
        all_rows: List[dict] = []
        for meta in get_active_datasets():
            if meta.task_category not in active_cats:
                continue
            if meta.hf_repo_id is None and meta.data_source not in _PROCESSORS:
                logger.info("Skipping %s (no HF repo and no processor)", meta.data_source)
                continue

            logger.info("Processing %s / %s ...", meta.data_source, split)
            rows = process_dataset(meta, split)
            if args.max_per_dataset and len(rows) > args.max_per_dataset:
                rows = rows[: args.max_per_dataset]
            logger.info("  -> %d rows", len(rows))
            all_rows.extend(rows)

        if not all_rows:
            logger.warning("No rows for split=%s", split)
            continue

        df = pd.DataFrame(all_rows)
        out_path = os.path.join(local_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
