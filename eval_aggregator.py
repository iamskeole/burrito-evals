from __future__ import annotations

import re
import json
import csv
from pathlib import Path
from typing import Any, Dict, List


# =========================
# CONFIG
# =========================

MODEL_NAME_ROOT = "gpt-oss-20b"

COLUMN_ORDER = [
    "run_name",
    "backend",
    "wire_api",
    "model_name",
    "fc_model",

    "test_id",
    "test_name",
    "test_type",

    "browser_enabled",
    "python_enabled",
    "batch_size",
    "seed",
    "reasoning_effort",
    "temperature",

    "input_token_count",
    "output_token_count",

    "reasoning_token_count",
    "response_token_count",

    "latency",
    "correct",
    "n_tool_calls_browser",
    "n_tool_calls_python",

    "mt_input_token_count_success",
    "mt_output_token_count_success",
    "mt_latency_success",

    "mt_num_turns_total",
    "mt_num_turns_success",
    "mt_num_turns_success_pct",

    "is_error",
    "error_type",
    "error_message",
    "failure_mode",
]

def safe_json_load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            lines = f.read().splitlines()
            loaded = []
            for l in lines:
                try:
                    loaded.append(json.loads(l))
                except Exception as e:
                    print(f"Error loading {l}:\n{e}")
                    continue
            return loaded
        except Exception as e:
            print(f"Error loading {path}:\n{e}")
            return None


class BaseAggregator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def map_result(self, entry: Dict, metadata: Dict) -> Dict[str, Any]:
        raise NotImplementedError

    def collect(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def write_csv(self, rows: List[Dict[str, Any]], out_path: Path):
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            # We use extrasaction='ignore' so internal tracking keys don't blow up the writer
            writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


class BFCLAggregator(BaseAggregator):
    # Matches: {backend}_{wire_api}_{reasoning}_t-{temperature}_b-{batch}_f-{fc}_s-{seed}
    _RUN_PATTERN = re.compile(
        r"^(?P<backend>[^_]+)_"
        r"(?P<wire_api>[^_]+)_"
        r"(?P<reasoning_effort>[^_]+)_"
        r"t-(?P<temperature>[^_]+)_"
        r"b-(?P<batch_size>\d+)_"
        r"f-(?P<fc_model>\d+)_"
        r"s-(?P<seed>-?\d+)$"  # Handles optional negative seeds
    )

    def _parse_run(self, dirname: str):
        match = self._RUN_PATTERN.match(dirname)
        if not match:
            raise ValueError(
                f"Directory name '{dirname}' does not match the expected "
                f"BFCLAggregator pattern."
            )

        gd = match.groupdict()

        return {
            "run_name": dirname,
            "backend": gd["backend"],
            "wire_api": gd["wire_api"],
            "reasoning_effort": gd["reasoning_effort"],
            "temperature": float(gd["temperature"]),
            "batch_size": int(gd["batch_size"]),
            "fc_model": int(gd["fc_model"]),
            "seed": int(gd["seed"]),
            "browser_enabled": 0,
            "python_enabled": 0,
            "is_error": 0,
            "n_tool_calls_browser": 0,
            "n_tool_calls_python": 0,
            "error_type": None,
            "error_message": None,

            "mt_input_token_count_success": 0,
            "mt_output_token_count_success": 0,
            "mt_latency_success": 0,

            "mt_num_turns_total": 0,
            "mt_num_turns_success": 0,
            "mt_num_turns_success_pct": 0.0,

            "reasoning_token_count": 0,
            "response_token_count": 0,
        }

    def map_stats_multi(self, entry, metadata):
        result = {
            "input_token_count": 0,
            "output_token_count": 0,
            "latency": 0,

            "mt_input_token_count_success": 0,
            "mt_output_token_count_success": 0,
            "mt_latency_success": 0,

            "mt_num_turns_total": 0,
            "mt_num_turns_success": 0,
            "mt_num_turns_success_pct": 0.0,
        }

        # 1. Determine if it was a complete success, or find the failure turn
        entry_error = entry.get("score", {}).get("error") if entry.get("score") else {}
        is_success = False if entry_error else True

        # Removed the early return for multi_turn:inference_error here!
        # Even if it force terminated, we still want to count the partial tokens/latency 
        # it consumed *before* it crashed.

        failed_turn_idx = None
        if not is_success:
            # The length of 'execution_result' tells us how many turns were evaluated before failing.
            # Length 1 = Failed on Turn 0 (Index 0). Length 2 = Failed on Turn 1 (Index 1).
            details = entry_error.get("details", {})
            exec_results = details.get("execution_result") or entry_error.get("execution_result", [])

            if exec_results:
                failed_turn_idx = len(exec_results) - 1
            else:
                # Fallback: Sometimes BFCL just puts the turn in the error message
                msg = str(entry_error.get("error_message", ""))
                match = re.search(r"turn (\d+)", msg, re.IGNORECASE)
                if match:
                    failed_turn_idx = int(match.group(1))
                else:
                    failed_turn_idx = 0 # Worst case fallback

        # 2. Extract nested metric lists from result payload
        entry_result = entry.get("result", {})
        input_tokens = entry_result.get("input_token_count", [])
        output_tokens = entry_result.get("output_token_count", [])
        latencies = entry_result.get("latency", [])

        total_turns = len(input_tokens)
        if total_turns == 0:
            return result # Failsafe for completely empty runs

        result["mt_num_turns_total"] = total_turns

        # 3. Iterate through turns and assign partial credit
        for turn_idx in range(total_turns):
            turn_in_toks = sum(input_tokens[turn_idx]) if isinstance(input_tokens[turn_idx], list) else input_tokens[turn_idx]
            turn_out_toks = sum(output_tokens[turn_idx]) if turn_idx < len(output_tokens) else 0
            turn_lat = sum(latencies[turn_idx]) if turn_idx < len(latencies) else 0

            # Always add to total overarching metrics
            result["input_token_count"] += turn_in_toks
            result["output_token_count"] += turn_out_toks
            result["latency"] += turn_lat

            # Add to success buckets ONLY IF this specific turn passed
            is_turn_successful = is_success or (failed_turn_idx is not None and turn_idx < failed_turn_idx)

            if is_turn_successful:
                result["mt_num_turns_success"] += 1
                result["mt_input_token_count_success"] += turn_in_toks
                result["mt_output_token_count_success"] += turn_out_toks
                result["mt_latency_success"] += turn_lat

        # 4. Calculate a percentage completion score (e.g. 50% if it passed 2 out of 4 turns)
        result["mt_num_turns_success_pct"] = (result["mt_num_turns_success"] / total_turns) * 100.0

        return result
    
    def map_error(self, entry, metadata):
        result = {
            "is_error": 0,
            "error_type": None,
            "error_message": None,
        }

        entry_score = entry["score"]
        has_hard_error = "traceback" in entry["result"]
        is_success = False if entry_score or has_hard_error else True
        if is_success:
            return result

        entry_error = entry_score["error"] if entry_score else {}

        if has_hard_error:
            err_str = entry["result"]["result"]
            if "'NoneType' object" in err_str:
                entry_error = {
                    "error_type": "server:probable_doom_loop",
                    "error_message": "Model likely entered an endless loop and exceded context window."
                }
            elif "unexpected tokens remaining in message header" in err_str.lower():
                entry_error = {
                    "error_type": "server:harmony_error_header",
                    "error_message": "Model issued token(s) server couldn't parse in Harmony header."
                }
            elif "failed to parse input at pos" in err_str.lower():
                entry_error = {
                    "error_type": "server:harmony_error_parser",
                    "error_message": "Model likely issued token(s) that break promise it also issued in <|constrain|>."
                }
            elif "unknown input type: " in err_str.lower():
                entry_error = {
                    "error_type": "server:bad_input_type",
                    "error_message": "Bad or malformed request payload reached the server. May or may not have been due to model hallucinations."
                }
            elif "unknown role: " in err_str.lower():
                entry_error = {
                    "error_type": "server:harmony_error_role",
                    "error_message": "Backend unable to process role token(s) issued by model."
                }
            elif "unsupported output object: " in err_str.lower():
                entry_error = {
                    "error_type": "server:bad_output_type",
                    "error_message": "Server tried to output unsupported Type object."
                }
            elif "unexpected token" in err_str.lower() and "while expecting start token" in err_str.lower():
                entry_error = {
                    "error_type": "server:harmony_error_token_sequence",
                    "error_message": "Server unable to process Harmony token sequence."
                }
            elif "unknown browser action" in err_str.lower():
                entry_error = {
                    "error_type": "server:fatal_type_hallucination",
                    "error_message": "Model hallucinated tool types backend refused."
                }
            else:
                entry_error = {
                    "error_type": "server:inference_error",
                    "error_message": "Unknown server error has occured."
                }
            x = 1

        elif isinstance(entry_error, list): # sometimes for single turn
            entry_error = {
                "error_type": entry["score"]["error_type"],
                "error_message": " | ".join(entry["score"]["error"])
            }

        error_type = entry_error["error_type"]
        error_message = entry_error["error_message"]

        # Build taxonomy
        if error_type in [
            "server:inference_error",
            "server:probable_doom_loop",
            "server:harmony_error_header",
            "server:harmony_error_parser",
            "server:harmony_error_role",
            "server:harmony_error_token_sequence",
            "server:fatal_type_hallucination",
            "server:bad_input_type",
            "server:bad_output_type",
            "agentic:inference_error",
            "multi_turn:inference_error",
            "multi_turn:force_terminated",
        ]:
            result["is_error"] = 1

        elif error_type == "multi_turn:empty_turn_model_response":
            result["is_error"] = 0
        else:
            result["is_error"] = 0

        result["error_type"] = error_type
        result["error_message"] = error_message
        return result
    
    def map_stats_single(self, entry, metadata):
        # We define the mt_ keys so that unpacking doesn't break, 
        # but logically a single turn either completely succeeds (1 turn passed) or fails.
        result = {
            "input_token_count": 0,
            "output_token_count": 0,
            "latency": 0,
            
            "mt_input_token_count_success": 0,
            "mt_output_token_count_success": 0,
            "mt_latency_success": 0,

            "mt_num_turns_total": 1,
            "mt_num_turns_success": 0,
            "mt_num_turns_success_pct": 0.0,
        }
        
        entry_result = entry.get("result", {})
        if isinstance(entry_result, dict):
            # Helper to extract safely (sometimes BFCL wraps these in lists even in single-turn)
            def _get_val(key):
                val = entry_result.get(key, 0)
                return sum(val) if isinstance(val, list) else val

            result["input_token_count"] = _get_val("input_token_count")
            result["output_token_count"] = _get_val("output_token_count")
            result["latency"] = _get_val("latency")

            # Check success condition for single turn
            if not entry.get("score"):
                result["mt_num_turns_success"] = 1
                result["mt_num_turns_success_pct"] = 100.0
                result["mt_input_token_count_success"] = result["input_token_count"]
                result["mt_output_token_count_success"] = result["output_token_count"]
                result["mt_latency_success"] = result["latency"]
                
        return result


    def map_result(self, entry, metadata):
        entry_score = entry.get("score")
        entry_result = entry.get("result")

        is_multi_turn = "multi_turn" in metadata["test_name"]

        if is_multi_turn:
            stats = self.map_stats_multi(entry, metadata)
            error = self.map_error(entry, metadata)
            ltc = stats["latency"]
            if not ltc:
                x = 1
                if not error["is_error"]:
                    x = 2
        else:
            stats = self.map_stats_single(entry, metadata)
            error = self.map_error(entry, metadata)
            ltc = stats["latency"]
            if not ltc:
                x = 1
                if not error["is_error"]:
                    retry = error = self.map_error(entry, metadata)
                    x = 2

        # Fallback ID extraction
        if entry_result and "id" in entry_result:
            tid = entry_result["id"]
        elif entry_score and "id" in entry_score:
            tid = entry_score["id"]
        else:
            tid = "unknown"

        correct = 1 if not entry_score else 0
        failure_mode = "success"

        if error["error_type"] is not None:
            failure_mode = "infra" if "server" in error["error_type"] else "model"

        # defensive where in some instances irrelevance flags even 
        # server errors as correct answers as they don't call a function
        if correct and error["is_error"] == 1:
            correct = 0
            failure_mode = "infra"

        row = {
            **metadata,
            "test_id": tid,
            "correct": correct,
            **stats, 
            **error,
            "failure_mode": failure_mode
        }
        return row

    def collect(self):
        rows = []
        root = self.base_dir / "bfcl_v4"

        if not root.exists():
            return rows

        for run_dir in root.iterdir():
            if not run_dir.is_dir() or "_discard" in run_dir.name:
                continue

            metadata = self._parse_run(run_dir.name)

            for result_file in run_dir.rglob("*_result.json"):
                # REMOVED: if "multi_turn" not in result_file.name: continue
                # We want to process single-turn files too!

                score_file = Path(
                    str(result_file)
                    .replace("/result/", "/score/")
                    .replace("_result.json", "_score.json")
                )

                model_name = MODEL_NAME_ROOT
                if metadata["fc_model"]:
                    model_name += "-FC"

                entry_meta = metadata.copy()
                entry_meta["model_name"] = model_name
                entry_meta["test_type"] = result_file.parent.name
                entry_meta["test_name"] = result_file.stem.replace("BFCL_v4_", "").replace("_result", "")

                score_data = safe_json_load(score_file)
                result_data = safe_json_load(result_file)

                if not score_data or not result_data:
                    continue

                # Ensure we are dealing with lists
                if isinstance(score_data, dict):
                    score_data = [score_data]
                if isinstance(result_data, dict):
                    result_data = [result_data]

                # STRICT FILTERING: Keep only dictionaries that contain an "id".
                # This drops garbage strings AND naturally removes the BFCL {"accuracy": ...} header!
                score_data = [i for i in score_data if isinstance(i, dict) and "id" in i]
                result_data = [i for i in result_data if isinstance(i, dict) and "id" in i]

                if not result_data:
                    print(f"Missing result data for {result_file.name}")
                    continue

                id_list = sorted(list(set([
                    i["id"] for i in score_data] + [i["id"] for i in result_data
                ])))

                entries = {i: {"score": None, "result": None} for i in id_list}
                
                for i in score_data:
                    entries[i["id"]]["score"] = i

                for i in result_data:
                    if entries[i["id"]]["result"] is None:
                        entries[i["id"]]["result"] = i

                for entry in entries.values():
                    row = self.map_result(entry, entry_meta)
                    rows.append(row)

        return rows


class GPTOSSAggregator(BaseAggregator):
    # Matches: {backend}_{wire_api}_{reasoning}_t-{temperature}_b-{batch}_be-{be}_pe-{pe}_s-{seed}
    _RUN_PATTERN = re.compile(
        r"^(?P<backend>[^_]+)_"
        r"(?P<wire_api>[^_]+)_"
        r"(?P<reasoning_effort>[^_]+)_"
        r"t-(?P<temperature>[^_]+)_"
        r"b-(?P<batch_size>\d+)_"
        r"be-(?P<browser_enabled>\d+)_"
        r"pe-(?P<python_enabled>\d+)_"
        r"s-(?P<seed>-?\d+)$"  # Handles optional negative seeds
    )

    def _parse_run(self, dirname: str):
        match = self._RUN_PATTERN.match(dirname)
        if not match:
            raise ValueError(
                f"Directory name '{dirname}' does not match the expected "
                f"GPTOSSAggregator pattern."
            )

        gd = match.groupdict()
        browser_enabled = int(gd["browser_enabled"])
        python_enabled = int(gd["python_enabled"])
        fc_model = 1 if (browser_enabled or python_enabled) else 0

        return {
            "run_name": dirname,
            "backend": gd["backend"],
            "wire_api": gd["wire_api"],
            "reasoning_effort": gd["reasoning_effort"],
            "temperature": float(gd["temperature"]),
            "batch_size": int(gd["batch_size"]),
            "browser_enabled": browser_enabled,
            "python_enabled": python_enabled,
            "seed": int(gd["seed"]),
            "fc_model": fc_model,
            "is_error": 0,
            "n_tool_calls_browser": 0,
            "n_tool_calls_python": 0,
            "error_type": None,
            "error_message": None,

            "mt_input_token_count_success": 0,
            "mt_output_token_count_success": 0,
            "mt_latency_success": 0,

            "mt_num_turns_total": 0,
            "mt_num_turns_success": 0,
            "mt_num_turns_success_pct": 0.0,
        }

    def collect(self):
        rows = []
        root = self.base_dir / "gpt_oss"
        
        if not root.exists():
            return rows

        for run_dir in root.iterdir():
            if not run_dir.is_dir() or "_discard" in run_dir.name:
                continue

            meta = self._parse_run(run_dir.name)

            for result_file in run_dir.rglob("*_allresults.json"):
                data = safe_json_load(result_file)
                if not isinstance(data, dict):
                    continue
                if "metadata" not in data:
                    continue

                test_results = data["metadata"]["example_level_metadata"]

                model_name = MODEL_NAME_ROOT
                if meta["fc_model"]:
                    model_name += "-FC"

                for obj in test_results:
                    correct = int(obj.get("correct", 0))
                    is_error = int(obj.get("n_errors", 0) > 0)

                    test_id = obj["test_id"]
                    test_name = test_id.split("_")[0].upper()
                    test_type = "GPT-OSS"

                    error_type, error_message, failure_mode = None, None, "success"
                    if is_error:
                        error_type = "server:probable_doom_loop"
                        error_message = "Model likely entered an endless loop and exceded context window."
                        failure_mode = "infra"

                    toks_in = obj.get("n_input_tokens", 0)
                    toks_out = obj.get("n_output_tokens", 0)
                    toks_reasoning = obj.get("n_reasoning_tokens", 0)
                    toks_response = obj.get("n_response_tokens", 0)
                    latency = obj.get("latency", 0)

                    row = {
                        **meta,
                        "model_name": model_name,
                        "test_type": test_type,
                        "test_id": test_id,
                        "test_name": test_name,
                        "input_token_count": toks_in,
                        "output_token_count": toks_out,
                        "latency": latency,
                        "correct": correct,
                        "n_tool_calls_browser": obj["n_tool_calls_browser"],
                        "n_tool_calls_python": obj["n_tool_calls_python"],
                        "is_error": is_error,
                        "error_type": error_type,
                        "error_message": error_message,
                        "failure_mode": failure_mode,

                        "reasoning_token_count": toks_reasoning,
                        "response_token_count": toks_response,

                        "mt_input_token_count_success": toks_in,
                        "mt_output_token_count_success": toks_out,
                        "mt_latency_success": latency,

                        "mt_num_turns_total": 1,
                        "mt_num_turns_success": correct,
                        "mt_num_turns_success_pct": correct,
                    }

                    rows.append(row)

        return rows


def main():
    base = Path(__file__).parent / "data"

    bfcl = BFCLAggregator(base)
    gptoss = GPTOSSAggregator(base)

    rows = []
    rows.extend(bfcl.collect())
    rows.extend(gptoss.collect())

    out_path = base / "eval_results.csv"

    bfcl.write_csv(rows, out_path)
    print(f"Aggregation complete. Wrote {len(rows)} rows to {out_path}.")


if __name__ == "__main__":
    main()