#!/usr/bin/env python
"""Scratch (read-only-analysis) script for the 3.49B-token headline composition.

Reads burrito-evals/data/eval_results.csv and reports:
  - total input tokens
  - total output tokens
  - within output: total reasoning/thinking tokens vs visible-answer tokens
  - overall total, plus sanity check against the headline 3.49B.
Writes the full-precision results to report/.polish-composition.md and prints a summary.
"""

import os

import pandas as pd

CSV = "/home/p/code/local/burrito-evals/data/eval_results.csv"
OUT_MD = "/home/p/code/local/burrito-evals/report/.polish-composition.md"

HEADLINE = 3_490_000_000  # post states total tokens across all runs = 3.49B

cols = [
    "backend",
    "input_token_count",
    "output_token_count",
    "reasoning_token_count",
    "response_token_count",
    "mt_input_token_count_success",
    "mt_output_token_count_success",
]

df = pd.read_csv(CSV, usecols=cols)
n_rows = len(df)

report = []
report.append("# 3.49B headline token composition (scratch)")
report.append("")
report.append(f"- Data file: `{CSV}`")
report.append(f"- Rows: {n_rows}")
report.append(f"- Columns used (verbatim from header): {', '.join(cols)}")
report.append("")
report.append("## Column semantics (by name)")
report.append("- `input_token_count`: input/prompt tokens")
report.append("- `output_token_count`: output tokens (model-generated total)")
report.append("- `reasoning_token_count`: reasoning/thinking tokens within output")
report.append("- `response_token_count`: visible-answer (non-reasoning) tokens within output")
report.append("- `mt_input_token_count_success` / `mt_output_token_count_success`: multi-turn success-only subset (auxiliary)")
report.append("")

# --- dtype / null diagnostics ---
report.append("## Dtypes and null counts")
report.append("")
report.append("| column | dtype | nulls |")
report.append("|---|---|---|")
for c in cols:
    report.append(f"| {c} | {df[c].dtype} | {int(df[c].isna().sum())} |")
report.append("")

# --- totals (sum of non-null values) ---
tot = {}
for c in ["input_token_count", "output_token_count", "reasoning_token_count", "response_token_count"]:
    tot[c] = int(df[c].sum(min_count=1)) if df[c].notna().any() else 0

mt_in = int(df["mt_input_token_count_success"].sum()) if df["mt_input_token_count_success"].notna().any() else 0
mt_out = int(df["mt_output_token_count_success"].sum()) if df["mt_output_token_count_success"].notna().any() else 0

# --- consistency checks on the reasoning/visible split ---
sum_split = df["reasoning_token_count"].fillna(0) + df["response_token_count"].fillna(0)
rows_split_eq = int((df["output_token_count"].fillna(0) == sum_split).sum())
rows_split_ne = n_rows - rows_split_eq
reasoning_max = int(df["reasoning_token_count"].max()) if df["reasoning_token_count"].notna().any() else 0
reasoning_nonzero = int((df["reasoning_token_count"].fillna(0) > 0).sum())
output_nonzero = int((df["output_token_count"].fillna(0) > 0).sum())

report.append("## Split consistency")
report.append(f"- rows where output_token_count == reasoning_token_count + response_token_count: {rows_split_eq} / {n_rows}")
report.append(f"- rows where the split does NOT sum to output_token_count: {rows_split_ne}")
report.append(f"- rows with reasoning_token_count > 0: {reasoning_nonzero}")
report.append(f"- max reasoning_token_count: {reasoning_max}")
report.append(f"- rows with output_token_count > 0: {output_nonzero}")
report.append("")

# --- headline candidates ---
total_in = tot["input_token_count"]
total_out = tot["output_token_count"]
total_reason = tot["reasoning_token_count"]
total_resp = tot["response_token_count"]
overall = total_in + total_out

cand_A = overall                      # input + output (reasoning inside output)
cand_B = total_in + total_reason + total_resp   # input + reasoning + visible (in case output != split)
cand_C = overall + total_reason       # if reasoning were additive on top of output (double-count hypothesis)

def pct(x):
    return (x - HEADLINE) / HEADLINE * 100.0

# --- per-backend characterisation of the split ---
by_backend = (
    df.groupby("backend", observed=True)
    .agg(
        rows=("output_token_count", "size"),
        output=("output_token_count", "sum"),
        reasoning=("reasoning_token_count", "sum"),
        response=("response_token_count", "sum"),
    )
    .reset_index()
)
report.append("## Per-backend split availability")
report.append("")
report.append("| backend | rows | output tokens | reasoning tokens | response tokens | unsplit output (out - reasoning - response) |")
report.append("|---|---|---|---|---|---|")
for _, r in by_backend.iterrows():
    unsplit = r["output"] - r["reasoning"] - r["response"]
    report.append(
        f"| {r['backend']} | {int(r['rows'])} | {int(r['output'])} | {int(r['reasoning'])} | {int(r['response'])} | {int(unsplit)} |"
    )
report.append("")

# --- refined visible estimate: unsplit output treated as visible answer ---
unsplit_output = total_out - total_reason - total_resp
visible_est = total_resp + max(unsplit_output, 0)
report.append(f"- output tokens with NO recorded reasoning/visible split (output > 0, reasoning = 0, response = 0): {unsplit_output}")
report.append(f"- estimated visible-answer total if unsplit output is all visible: {visible_est}")
report.append("")

report.append("## Computed totals (full precision)")
report.append("")
report.append(f"| quantity | tokens |")
report.append(f"|---|---|")
report.append(f"| total input (`input_token_count`) | {total_in} |")
report.append(f"| total output (`output_token_count`) | {total_out} |")
report.append(f"| total reasoning/thinking within output (`reasoning_token_count`) | {total_reason} |")
report.append(f"| total visible-answer within output (`response_token_count`) | {total_resp} |")
report.append(f"| overall (input + output) | {overall} |")
report.append(f"| aux: multi-turn success-only input (`mt_input_token_count_success`) | {mt_in} |")
report.append(f"| aux: multi-turn success-only output (`mt_output_token_count_success`) | {mt_out} |")
report.append("")
report.append("## Headline sanity check (3.49B = 3,490,000,000)")
report.append("")
report.append(f"| candidate definition | tokens | diff vs 3.49B | pct diff |")
report.append(f"|---|---|---|---|")
for label, v in [("A: input + output", cand_A), ("B: input + reasoning + visible", cand_B), ("C: input + output + reasoning (double-count)", cand_C)]:
    d = v - HEADLINE
    report.append(f"| {label} | {v} | {d:+d} | {pct(v):+.4f}% |")
report.append("")

best = min([("A: input + output", cand_A), ("B: input + reasoning + visible", cand_B), ("C: input + output + reasoning (double-count)", cand_C)], key=lambda kv: abs(kv[1] - HEADLINE))
verdict_ok = abs(pct(best[1])) <= 2.0
if verdict_ok:
    verdict = f"MATCH (within 2%): best candidate {best[0]} = {best[1]} tokens, {pct(best[1]):+.4f}% vs headline 3.49B."
else:
    verdict = f"MATERIAL INCONSISTENCY (>2%): best candidate {best[0]} = {best[1]} tokens, {pct(best[1]):+.4f}% vs headline 3.49B."
report.append(f"## Verdict")
report.append(f"- {verdict}")
report.append("")

os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
with open(OUT_MD, "w") as f:
    f.write("\n".join(report) + "\n")

print(f"rows={n_rows}")
print(f"input={total_in}")
print(f"output={total_out}")
print(f"reasoning={total_reason}")
print(f"response={total_resp}")
print(f"overall={overall}")
print(f"split_eq_rows={rows_split_eq}/{n_rows}")
print(f"headline=3.49B; A={cand_A} ({pct(cand_A):+.4f}%), B={cand_B} ({pct(cand_B):+.4f}%), C={cand_C} ({pct(cand_C):+.4f}%)")
print(f"verdict: {verdict}")
print(f"wrote {OUT_MD}")
