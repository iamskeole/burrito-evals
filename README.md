# burrito-evals

The evaluation suite and complete dataset for gpt-oss: 320,192 runs, 3.49B tokens, 1,062 GPU hours on a single RTX 3090.

## What

This is the public record behind the claims in [burrito-core](https://github.com/iamskeole/burrito-core) — the inference harness for gpt-oss, also MIT-licensed. That README summarizes the findings; this repo holds everything that produced them:

- **`data/eval_results.csv`** — the dataset: 320,192 rows, one per run, with backend, wire API, tool-schema mode, tool flags, reasoning effort, seed, token counts (input / output / reasoning / response), latency, correctness, tool-call counts, per-turn multi-turn bookkeeping, and error / failure mode.
- **`data/gpt_oss/`, `data/bfcl_v4/`** — the raw per-run model outputs including full reasoning traces (one directory per run configuration and seed). The `__discard_*` directories hold runs that were excluded from every reported number (bot-walled browser hits, a wrong-effort batch) and are kept for transparency. `report/dataset.md` documents the exact layout.
- **`artifacts/`** — both jinja chat templates exactly as served (`chat_template.jinja` default, `chat_template_fixed.jinja` the fix), and `eval_specs.xlsx`: the specification of all 320,192 runs.
- **`report/`** — `report.md` (the full report, seven findings), `post.md` (the public write-up), `story.md` (the reasoning-effort deep dive), `plots.md` (the canonical record of all 44 figures), `dataset.md` (the data reference: what each row is, every column, the phases, and the raw traces on disk), plus `sections/` (the report split per section) and planning notes.
- **`plots/`** — the 44 figures: five reliability metrics per phase (mean accuracy, pass@8, pass^8, fail^8, error rate).
- **The code** — `eval_aggregator.py` (the scoring/aggregation engine that produced the CSV; documents the column semantics, including multi-turn failure bookkeeping), `eval_helpers.py` (the analysis and plotting suite), `eval_report.ipynb` (regenerates every figure directly from the CSV), and `scripts/` (the exact serve configuration for each backend, plus the 3090 power-management service used during the run).

## What we found

- **Configuration matters more than model capability.** The same weights score from ~0% to ~87% depending on template, tool format, wire API, and effort level.
- **The default jinja template breaks the model.** Removing `commentary` from the valid output channels when no tools are present moves live-test accuracy from ~3% to ~40%; the fixed template puts vanilla llama.cpp at parity with the harness on tool-free benchmarks.
- **Wire API matters.** On vLLM, `/v1/responses` errors on 73.5–83.5% of multi-turn runs versus 29.0–36.0% on `/v1/chat/completions`.
- **Structured tool schemas (fc_model=1) triple multi-turn accuracy** over AST parsing (fc_model=0), where the vanilla backends collapse to ~0%.
- **Reasoning effort changes answer quality, not just length.** At a fixed ~1.4k reasoning-token budget, AIME25 accuracy is 38/97/100% for low/medium/high effort.
- **Native python tools add 21–24 points at low effort and 10–12 at medium**, while reducing token usage.

Full detail, all 44 figures, and per-run traces: [`report/report.md`](report/report.md).

## How it was run

- **Model and hardware** — gpt-oss-20b at factory MXFP4 over 128K context, on a single RTX 3090 (power-managed via `scripts/nvidia-power.service`); 1,062 GPU hours total, batch size 1, temperature 1.0. Versions are pinned in [`eval_versions.txt`](eval_versions.txt): llama.cpp commit `3a479c913`, vLLM `v0.21.0`, with model date strings frozen at 2026-05-25. The backends are May 2026 releases evaluated against the original August 2025 model weights.
- **Scale** — 320,192 runs = 40,024 (configuration, problem) pairs × 8 seeds. Each pair was run once per seed from a fixed seed set: `1337`, `24601`, `271828`, `680867`, `819437`, `4201337`, `10203040`, `314159265`.
- **Why 8 seeds** — every phase reports five metrics, and several of them are only defined over multiple independent attempts: mean accuracy, **pass@8** (solved by at least one of the eight seeds), **pass^8** (solved by all eight), **fail^8** (failed by all eight), and error rate. The triplet decomposes each task into reliably solved, reliably broken, and per-seed luck — e.g. tool-free AIME25 is 93.3% solvable by some seed but only 36.7% by all eight.
- **Backends (seven)** — burrito over llama.cpp and vLLM, each with and without preserved thinking (the `burrito-pt` variants), plus vanilla llama.cpp with the default and the fixed jinja template, and vanilla vLLM.
- **Benchmarks** — BFCL v4 in three modes: non-live (simple python/java/javascript, irrelevance), live (live relevance/simple, with a real browser), and `multi_turn_base` (agentic, 1-7 turns per task); plus AIME25 and GPQA from OpenAI's GPT-OSS reasoning suite.
- **Axes** — each backend was run on both `/v1/chat/completions` and `/v1/responses` (where applicable) with both tool-definition modes: `fc_model=1` (structured schemas) and `fc_model=0` (AST parsing, no tools in the request); low/medium/high reasoning effort; and, where a tool is relevant, with and without `python` / `browser` enabled.

## Reproducing

**The analysis needs no GPU** — the entire report derives from the CSV:

```bash
uv sync
# open eval_report.ipynb (or run its cells) — it recomputes the headline
# numbers (tokens, runs, seeds, GPU hours) straight from data/eval_results.csv
# and regenerates all 44 figures in plots/
```

`eval_helpers.py` contains the plotting suite (`EvalPlotter`; layout and export polish iterated with Grok 4.5), and `report/scripts/explore_data.py` is the exploration script behind the key data findings.

**Re-running the evals themselves** needs the GPU and a backend: `scripts/` holds the exact serve configuration for each of the seven backends (default vs fixed jinja for llama.cpp, default vs production for vLLM), and `artifacts/eval_specs.xlsx` enumerates every run.

## License

MIT-licensed, including all data, traces, and artifacts in this repo ([LICENSE](/LICENSE)).
