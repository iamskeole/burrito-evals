"""
Abstract and concrete aggregators for BFCL and GPT‑OSS evals.

This module contains all logic required to aggregate the two benchmark
formats into CSV files.  The current implementation focuses on the
BFCL workflow – GPT‑OSS parsing is left as a stub ready for later
implementation.
"""

# Import the built‑in ``Any`` type under a convenient name for type
# hints.  The original file did the same, but the alias was unused.
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any  # lol workaround for not managing to replace 'any' with 'Any'

# ----------------------------------------------------------------------
# Column definition
# ----------------------------------------------------------------------
# All aggregators share the same logical column order.  The list is
# defined once at the top of the file so changes propagate to both
# types.  ``browser_enabled`` and ``python_enabled`` are positioned next
# to ``fc_model`` as they are conceptually related.
COLUMN_ORDER: list[str] = [
    "run_name",
    "backend",
    "wire_api",
    "model_name",
    "fc_model",
    "browser_enabled",
    "python_enabled",
    "batch_size",
    "seed",
    "reasoning_effort",
    "temperature",
    "test_id",
    "test_name",
    "test_type",
    "input_token_count",
    "output_token_count",
    "latency",
    "correct",
]


# ----------------------------------------------------------------------
# Base aggregation logic
# ----------------------------------------------------------------------
class EvalAggregator:
    """Base class that implements the common aggregation pipeline.

    Sub‑classes only need to provide:
    * ``name`` – identifier used for the output file name.
    * ``column_order`` – optional explicit order for the CSV columns.
    * ``parse_run_dir`` – given a run folder name, return a dict of
      metadata common to each row.
    * ``collect_rows`` – walk the run directory tree and return a list of
      row dictionaries.
    """

    #: identifier used in the output file name (e.g. ``bfcl_v4``)
    name: str
    #: optional ordering of columns – defaults to ``COLUMN_ORDER``
    column_order: list[str] | None = None

    def __init__(self, base: Path):
        self.base = base
        self.out_csv = base / f"eval_results_{self.name}.csv"

    # ------------------------------------------------------------------
    # Methods that concrete classes must override
    # ------------------------------------------------------------------
    def parse_run_dir(self, dirname: str):
        raise NotImplementedError

    def collect_rows(self):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Common implementation
    # ------------------------------------------------------------------
    def write_csv(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            print("No data collected.", flush=True)
            return
        # Compute the union of all keys
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        headers = COLUMN_ORDER
        if self.column_order:
            col_order = self.column_order
            headers = sorted(
                headers,
                key=lambda k: (
                    col_order.index(k) if k in col_order else len(col_order),
                    k,
                ),
            )
        with open(self.out_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Aggregated {len(rows)} rows to {self.out_csv}", flush=True)


# ----------------------------------------------------------------------
# Concrete BFCL implementation
# ----------------------------------------------------------------------
class BFCLAggregator(EvalAggregator):
    name = "bfcl_v4"
    column_order = COLUMN_ORDER

    BFCL_DIR = None  # will be set in __init__

    def __init__(self, base: Path):
        super().__init__(base)
        self.BFCL_DIR = base / "bfcl_v4"

    def parse_run_dir(self, dirname: str):
        # Naming convention
        # ``{backend}_{wire_api}_{reasoning}_t-{temperature}_b-{batch}_f-{fc}_s-{seed}``
        parts = dirname.split("_")
        backend = parts[0]
        wire_api = parts[1]
        reasoning_effort = parts[2]
        temperature = parts[3].split("-")[1]
        batch_size = int(parts[4].split("-")[1])
        fc_model = int(parts[5].split("-")[1])
        seed = int(parts[6].split("-")[1])
        return {
            "run_name": dirname,
            "backend": backend,
            "wire_api": wire_api,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "batch_size": batch_size,
            "fc_model": fc_model,
            "seed": seed,
            "browser_enabled": 0,
            "python_enabled": 0,
        }

    def collect_rows(self):
        data_rows: list[dict[str, Any]] = []
        if self.BFCL_DIR is None:
            return data_rows
        for run_path in self.BFCL_DIR.iterdir():
            if not run_path.is_dir():
                continue
            meta = self.parse_run_dir(run_path.name)
            result_root = run_path / "result"
            if not result_root.exists():
                continue
            for model_path in result_root.iterdir():
                if not model_path.is_dir():
                    continue
                model_name = model_path.name
                for test_type_path in model_path.iterdir():
                    if not test_type_path.is_dir():
                        continue
                    test_type = test_type_path.name
                    score_root = run_path / "score" / model_name / test_type
                    for json_file in test_type_path.glob("*_result.json"):
                        score_file = (
                            score_root
                            / f"{json_file.name.replace('_result.json', '_score.json')}"
                        )
                        test_validity: dict[str, bool] = {}
                        if score_file.exists():
                            with open(score_file, "r", encoding="utf-8") as sf:
                                for line_value in sf:
                                    if not line_value.strip():
                                        continue
                                    try:
                                        entry = json.loads(line_value)
                                    except Exception:
                                        continue
                                    if isinstance(entry, dict) and "id" in entry:
                                        test_validity[entry["id"]] = False
                        with open(json_file, "r", encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                obj = json.loads(line)
                                # Extract numeric metrics, supporting nested under 'stats'
                                for key in [
                                    "input_token_count",
                                    "output_token_count",
                                    "latency",
                                ]:
                                    value = obj.get(key)
                                    if (
                                        value is None
                                        and "stats" in obj
                                        and isinstance(obj["stats"], dict)
                                    ):
                                        value = obj["stats"].get(key)
                                    if isinstance(value, list):
                                        total = sum(
                                            inner
                                            if isinstance(inner, int)
                                            else sum(inner)
                                            for inner in value
                                            if isinstance(inner, (list, int))
                                        )
                                        obj[key] = total
                                    else:
                                        # Keep original if not list
                                        obj[key] = (
                                            value if value is not None else obj.get(key)
                                        )
                                obj["correct"] = (
                                    0 if obj.get("id") in test_validity else 1
                                )
                                obj.pop("result", None)
                                test_id = obj.get("id", "")
                                test_name = (
                                    test_id.rsplit("_", 1)[0]
                                    if isinstance(test_id, str)
                                    else test_id
                                )
                                obj.pop("id", None)
                                # Keep only expected columns from obj
                                obj = {
                                    k: v for k, v in obj.items() if k in COLUMN_ORDER
                                }
                                row = {
                                    **meta,
                                    "model_name": model_name,
                                    "test_type": test_type,
                                    **obj,
                                    "test_id": test_id,
                                    "test_name": test_name,
                                }
                                data_rows.append(row)
        return data_rows


# ----------------------------------------------------------------------
# Placeholder GPT‑OSS implementation
# ----------------------------------------------------------------------
class GPTOSSAggregator(EvalAggregator):
    name = "gpt_oss"
    column_order = COLUMN_ORDER

    def __init__(self, base: Path):
        super().__init__(base)
        self.base_dir = base  # the entire repo – we walk from ``eval_type`` subdir

    def parse_run_dir(self, dirname: str):
        # Naming format:
        # ``{backend}_{wire_api}_{reasoning}_t-{temperature}_b-{batch}_be-{be}_pe-{pe}_s-{seed}``
        parts = dirname.split("_")
        backend = parts[0]
        wire_api = parts[1]
        reasoning_effort = parts[2]
        temperature = parts[3].split("-")[1]
        batch_size = int(parts[4].split("-")[1])
        browser_enabled = int(parts[5].split("-")[1])
        python_enabled = int(parts[6].split("-")[1])
        seed = int(parts[7].split("-")[1])
        return {
            "run_name": dirname,
            "backend": backend,
            "wire_api": wire_api,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "batch_size": batch_size,
            "browser_enabled": browser_enabled,
            "python_enabled": python_enabled,
            "seed": seed,
            "fc_model": 1,  # as per requirement
        }

    def collect_rows(self):
        rows: list[dict[str, Any]] = []
        # Base directory expected to contain a subfolder "gpt_oss" with run directories
        gpt_oss_path = self.base_dir / "gpt_oss"
        if not gpt_oss_path.exists() or not gpt_oss_path.is_dir():
            return rows
        for run_dir in gpt_oss_path.iterdir():
            if not run_dir.is_dir():
                continue
            # Identify the allresults.json file(s) for this run
            allresults_files = list(run_dir.glob("*_allresults.json"))
            if not allresults_files:
                continue
            # Parse common metadata from directory name
            meta = self.parse_run_dir(run_dir.name)
            for json_file in allresults_files:
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                    examples = data.get("metadata", {}).get(
                        "example_level_metadata", []
                    )
                    for ex in examples:
                        if not isinstance(ex, dict):
                            continue
                        row = dict(meta)
                        # Basic identifiers
                        row["test_id"] = ex.get("test_id")
                        # Token and latency information
                        row["input_token_count"] = ex.get(
                            "input_token_count"
                        ) or ex.get("n_input_tokens")
                        row["output_token_count"] = ex.get(
                            "output_token_count"
                        ) or ex.get("n_output_tokens")
                        row["latency"] = ex.get("latency")
                        # Accuracy flag
                        row["correct"] = ex.get("correct")
                        # Include a simple test_type derived from the filename prefix
                        stem = json_file.stem
                        if "_" in stem:
                            row["test_type"] = "gpt_oss"
                        test_id_val = ex.get("test_id")
                        row["test_id"] = test_id_val
                        row["test_name"] = (
                            test_id_val.split("_", 1)[0]
                            if isinstance(test_id_val, str)
                            else test_id_val
                        )
                        # Determine model_name based on wire_api
                        wire_api_val = meta.get("wire_api", "")
                        # Always include '-FC' suffix according to wire_api
                        if wire_api_val == "chat":
                            model_name_val = "gpt-oss-20b-chat-FC"
                        else:
                            model_name_val = "gpt-oss-20b-responses-FC"
                        row["model_name"] = model_name_val
                        # Keep only expected columns
                        filtered_row = {
                            k: v for k, v in row.items() if k in COLUMN_ORDER
                        }
                        rows.append(filtered_row)
                except Exception:
                    # If any issue reading/parsing this file, skip it
                    continue
        return rows


# ----------------------------------------------------------------------
# Factory helper
# ----------------------------------------------------------------------


def get_aggregator(eval_type: str, base: Path) -> EvalAggregator:
    mapping = {
        "bfcl_v4": BFCLAggregator,
        "gpt_oss": GPTOSSAggregator,
    }
    cls = mapping.get(eval_type)
    if cls is None:
        raise ValueError(f"Unknown eval type: {eval_type}")
    return cls(base)


# ----------------------------------------------------------------------
# Example CLI usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python aggregator.py bfcl_v4|gpt_oss|all", flush=True)
        exit(1)
    eval_name = sys.argv[1]
    base = Path(__file__).parent / "data"
    os.makedirs(base, exist_ok=True)
    if eval_name == "all":
        rows: list[dict[str, Any]] = []
        for t in ["bfcl_v4", "gpt_oss"]:
            agg = get_aggregator(t, base)
            rows.extend(agg.collect_rows())
        out_csv = base / "eval_results_all.csv"
        if not rows:
            print("No data collected.", flush=True)
        else:
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())
            headers = COLUMN_ORDER
            with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            print(f"Aggregated {len(rows)} rows to {out_csv}", flush=True)
    else:
        aggregator = get_aggregator(eval_name, base)
        rows = aggregator.collect_rows()
        aggregator.write_csv(rows)
