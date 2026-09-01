# The Dataset: `data/eval_results.csv`

The complete dataset for the gpt-oss-20b evaluation: **320,192 rows, one row per run**. A *run* is one model attempt at one benchmark item (a *unit*) under one configuration and one seed. Every number in [`report.md`](report.md), [`story.md`](story.md), and [`post.md`](post.md), and all 44 figures in [`../plots/`](../plots/), derive from this single file. No GPU is needed to work with it: [`../eval_report.ipynb`](../eval_report.ipynb) regenerates the entire report from the CSV alone.

Model and hardware: gpt-oss-20b at factory MXFP4 over 131K context, on a single RTX 3090, batch size 1, temperature 1.0, 1,062 GPU hours. Totals: 3,122,239,084 input tokens, 367,392,611 output tokens (3.49B combined).

## 1. Quick start

```python
import pandas as pd
df = pd.read_csv("data/eval_results.csv")

# any subset you want, e.g. the reasoning-effort story on AIME25
df.query("test_name == 'AIME25' and backend.str.contains('burrito') and wire_api == 'responses'")
```

- [`../eval_report.ipynb`](../eval_report.ipynb) reproduces every figure; cells 1 to 11 correspond to phases 1 to 6 (section 5). Each phase cell applies one filter and calls the plotting suite for that phase's figures.
- [`../eval_helpers.py`](../eval_helpers.py) holds the plotting suite (`EvalPlotter`). Every plot method takes the CSV dataframe plus a pandas `.query()` string and returns `(fig, ax, data)`, where `data` is the table actually drawn, so you can work from raw numbers rather than images. `apply_sql_filter()` is the query wrapper, and `pass_at_k`, `pass_hat_k`, `binomial_se` are the metric formulas.
- [`../eval_aggregator.py`](../eval_aggregator.py) is the code that produced this CSV from the raw directories (section 7). It documents the column semantics, including the multi-turn bookkeeping.
- [`scripts/explore_data.py`](scripts/explore_data.py) prints the key statistics behind each finding, straight from the CSV.

## 2. What a row is

320,192 runs = **40,024 (configuration, unit) pairs, each run once under each of 8 seeds**. The seed set is fixed: `1337`, `24601`, `271828`, `680867`, `819437`, `4201337`, `10203040`, `314159265`. Every row carries exactly one of the eight, and each of the 40,024 pairs has all eight.

Temperature 1.0 and batch size 1 on every row, deliberately: 1.0 is the producer-recommended sampling setting, and the goal was a realistic deployment configuration, not a tuned one. The eight seeds exist so that per-phase metrics like **pass@8** (solved by at least one of the eight seeds), **pass^8** (solved by all eight), and **fail^8** (failed by all eight) are defined, alongside mean accuracy and error rate.

## 3. Configuration axes

| Column | Values | Notes |
|---|---|---|
| `backend` | `burrito@llamacpp` (87,088 rows), `burrito@vllm` (87,088), `vllm` (42,272), `llamacpp@default-jinja` (42,272), `llamacpp@fixed-jinja` (42,272), `burrito-pt@llamacpp` (9,600), `burrito-pt@vllm` (9,600) | Burrito is the [burrito-core](https://github.com/iamskeole/burrito-core) harness over llama.cpp (GGUF) or vLLM (safetensors, MXFP4). The `-pt` variants preserve the full thinking history instead of pruning it. `llamacpp@fixed-jinja` is vanilla llama.cpp with the one-line jinja template fix (see `../artifacts/chat_template_fixed.jinja`). |
| `wire_api` | `responses` (219,072 rows), `chat` (101,120) | The client-facing endpoint: `/v1/responses` vs `/v1/chat/completions`. Burrito always speaks `/v1/completions` to the backend; the recorded value is the client-facing API, which is why burrito rows appear under both wires. |
| `fc_model` | `0` (167,584 rows), `1` (152,608) | 0: no tools in the request; the model's tool calls are parsed out of its plain text output (AST parsing). 1: structured tool schemas sent in the request. |
| `reasoning_effort` | `medium` (218,240), `low` (50,976), `high` (50,976) | The model's internal reasoning-depth switch. Low/medium/high turns out to change *what* the model does, not just how much (the subject of `story.md`). |
| `python_enabled` | `0` (318,752), `1` (1,440) | The model-native python interpreter. See section 8: this axis only exists on AIME25 with burrito and `fc_model=1`. |
| `browser_enabled` | `0` (all rows) | The model-native browser. Never exercised: the browser test campaign was killed mid-run (section 8). |
| `model_name` | `gpt-oss-20b` (167,584), `gpt-oss-20b-FC` (152,608) | Not a free axis: it mirrors `fc_model` (0 maps to `gpt-oss-20b`, 1 to `gpt-oss-20b-FC`, the function-calling variant as served). Use `fc_model` in analysis; the report prose just says gpt-oss-20b. |
| `batch_size`, `temperature` | always `1`, always `1.0` | Constants, kept in the CSV for completeness. |

## 4. Benchmarks

Nine tests in four categories, 1,392 units in total:

| Category (`test_type`) | `test_name` | Units | Rows | What it measures |
|---|---|---|---|---|
| BFCL v4 non-live | `simple_python` | 400 | 89,600 | Single-turn tool calls, python-flavored |
| | `simple_java` | 100 | 22,400 | Single-turn tool calls, java-flavored |
| | `simple_javascript` | 50 | 11,200 | Single-turn tool calls, js-flavored |
| | `irrelevance` | 240 | 53,760 | The model must produce *no* tool call when none is relevant |
| BFCL v4 live | `live_simple` | 258 | 57,792 | Straightforward calls against live tool servers |
| | `live_relevance` | 16 | 3,584 | Deciding which of several live tools apply |
| BFCL v4 multi-turn | `multi_turn_base` | 200 | 64,000 | Agentic multi-step tasks, 1 to 7 turns each, scored against instance state |
| GPT-OSS native suite | `AIME25` | 30 | 3,600 | Math problem solving (OpenAI's GPT-OSS reasoning suite) |
| | `GPQA` | 198 | 14,256 | Graduate-level science (GPQA Diamond) |

BFCL v4 is the Berkeley Function Calling Leaderboard; the live and multi-turn categories run against the BFCL-provided tool servers. `test_id` identifies the unit: sequential indices for most BFCL tests (`simple_python_0` to `simple_python_99`, `multi_turn_base_0` to `multi_turn_base_199`), UUID-based ids for the GPT-OSS suite (`aime25_0cd2e8e5-...`), and `live_simple_0-0-0`-style composite ids for the live tests. The v4 dataset and its expected outputs are provided by the BFCL evaluation library at run time and are not committed here; the raw per-task outputs and score summaries they produced are (section 7).

## 5. Phases

The data was collected in numbered sweeps, one per phase: a fixed choice of backend group, wire API, and effort scope. There is **no stored phase column**; a row's phase is derivable from its axes, and `plots.md` states the exact scope of each phase. The phases are:

| Phase | Scope as run | Filter used by `eval_report.ipynb` |
|---|---|---|
| 1 | Full sweep at `medium` effort: all BFCL tests, the five non-preserved-thinking backends, both wire APIs, both `fc_model` modes | `reasoning_effort == 'medium' and test_type != 'GPT-OSS' and not backend.str.contains('burrito-pt')` |
| 2 | GPT-OSS baseline: AIME25 and GPQA, five non-pt backends, responses wire, `fc_model=0`, no tools | `reasoning_effort == 'medium' and test_type == 'GPT-OSS' and not backend.str.contains('burrito-pt') and python_enabled == 0` |
| 3 | BFCL reasoning-effort sweep: burrito backends (both), responses wire, low/medium/high, both fc modes | `test_type != 'GPT-OSS' and wire_api == 'responses' and backend.str.contains('burrito') and not backend.str.contains('burrito-pt')` |
| 4 | GPT-OSS effort sweep: burrito, responses wire, low/medium/high; AIME25 with python, GPQA without tools | `test_type == 'GPT-OSS' and wire_api == 'responses' and backend.str.contains('burrito') and not backend.str.contains('burrito-pt')` |
| 5 | `multi_turn_base` deep dive: all seven backends including the two preserved-thinking variants, all efforts, both fc modes | `test_name == 'multi_turn_base'` |
| 6 | Exploratory reasoning-effort study: **no new runs**, a re-analysis of existing rows (token binning, matched-length controls, effect/cost/tradeoff cuts) | `wire_api == 'responses'` plus per-figure scoping |

Only phase 1 is a matched cross-product; once the baseline was established, later phases dropped the chat wire and other combinations to use the single 3090 efficiently. Phases 5 and 6 are where the preserved-thinking variants appear and where the reasoning story is built. Concretely, phase 1 accounts for 25,280 of the 40,024 configuration-unit pairs, phase 2 for 1,140, phase 3 for 10,112 new pairs, phase 4 for 1,092, and phase 5 for 2,400; phase 6 adds zero.

## 6. Column reference

All 33 columns, in file order.

**Identity**

| Column | Meaning |
|---|---|
| `run_name` | Unique name of the run's configuration: `<backend>_<wire>_<effort>_t-<temp>_b-<batch>_<f-<fc> or be-<browser>_pe-<python>>_s-<seed>`, e.g. `burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867`. BFCL configurations carry the `_f-` tool-mode token; GPT-OSS configurations carry `_be-_pe-` tool flags. It also names the raw-data directory on disk (section 7). |
| `backend`, `wire_api`, `model_name`, `fc_model` | See section 3. |
| `test_id` | The benchmark unit (one row set per unit, per configuration, per seed). |
| `test_name`, `test_type` | The test and its category (section 4). |

**Configuration**

`browser_enabled`, `python_enabled`, `batch_size`, `seed`, `reasoning_effort`, `temperature` as in section 3.

**Tokens and latency**

| Column | Meaning |
|---|---|
| `input_token_count` | Total input tokens for the run (cumulative across turns on multi-turn runs). 3.12B across the dataset. |
| `output_token_count` | Total output tokens for the run, the whole: thinking plus visible answer. 367M across the dataset. This is the authoritative total. |
| `reasoning_token_count` | The thinking portion of output. Recorded only for GPT-OSS runs served on the responses wire (14,124 rows: 11,083 GPQA, 3,041 AIME25; 6,151 `burrito@vllm`, 6,149 `burrito@llamacpp`, 1,824 vanilla `vllm`). Zero everywhere else, including all vanilla llama.cpp runs, which record no thinking split at all. |
| `response_token_count` | The visible-answer portion of output, same coverage (17,711 rows). Wherever both split parts are recorded (14,063 rows) the identity `output = reasoning + response` holds exactly, as do the degenerate cases (61 rows reasoning-only, 3,648 response-only). |
| `latency` | Wall-clock seconds for the whole run. Median 1.37 s; the maximum (3,133.6 s) is a long multi-turn run. Always at least `mt_latency_success`. |

**Correctness and tools**

| Column | Meaning |
|---|---|
| `correct` | 0/1 as scored by the benchmark. 192,621 of 320,192 (60.2%). |
| `n_tool_calls_browser` | Browser tool calls made in the run. Always 0 (section 8). |
| `n_tool_calls_python` | Python tool calls made in the run. Nonzero only on the python-enabled AIME25 rows (12,458 total, up to 112 in a single run). |

**Multi-turn bookkeeping**

| Column | Meaning |
|---|---|
| `mt_num_turns_total` | Turns the run executed. On every single-turn test this is 1. On `multi_turn_base` (tasks define 1 to 7 turns) it is the number of turns actually attempted, and 0 for the 3,800 runs that never got one off the ground (all infrastructure errors). |
| `mt_num_turns_success` | Turns completed without failure, 0 to total. On single-turn rows it equals `correct`. |
| `mt_num_turns_success_pct` | Success turns over total, times 100 (0 when total is 0). |
| `mt_failed_turn_idx` | 0-indexed index of the first turn that failed. NaN if and only if the run is correct. On the 127,571 failed runs the first failure sits at turn 0 for 113,647 (89%), at turn 1 for 7,178, and spreads out to turn 6 (44 runs). |
| `mt_input_token_count_success`, `mt_output_token_count_success`, `mt_latency_success` | Sums over the successful turns only. On single-turn rows they mirror the run totals when the run is correct, and are 0 otherwise; on multi-turn rows they count only the turns that cleared, so a run that died at turn 3 of 5 still carries the first three turns' tokens. |

**Errors and failure modes**

| Column | Meaning |
|---|---|
| `is_error` | 1 for hard failures: 10,952 infrastructure errors plus 71 `multi_turn:force_terminated` (11,023 total). Wrong model answers do *not* set it; they stay `is_error=0` with a `failure_mode` of `model` (or `success`). |
| `error_type` | Machine-readable cause, set on the 122,044 rows with a diagnosable failure. Not set on the 5,527 "clean" wrong answers (the model produced a normal answer that scored wrong) nor on any correct row. `error_message` (human-readable detail) is set exactly where `error_type` is. |
| `failure_mode` | `success` (198,148): the run completed and produced an answer, whether or not it scored. `model` (111,092): the model produced an unusable output (empty turn, malformed decode, doom loop) or a wrong call. `infra` (10,952): the backend or harness failed to serve the request (server errors, timeouts); every one of these is `is_error=1`. |

The `error_type` values, with counts, grouped by what went wrong:

| Group | Types (count) |
|---|---|
| Decode failures on the AST path (`fc_model=0`) | `ast_decoder:decoder_failed` (23,688), `ast_decoder:decoder_wrong_output_format` (2,517) |
| Wrong tool call, scorer's verdict | `simple_function_checker:wrong_count` (15,840), `wrong_func_name` (871), `missing_required` (780), `missing_optional` (569), `unexpected_param` (188) |
| Decoded but wrong argument value/type | `value_error:string` (9,102), `type_error:simple` (5,262), `value_error:others` (2,120), `value_error:list/tuple` (1,400), `value_error:dict_key` (980), `type_error:nested` (833), `value_error:dict_value` (256), `type_error:js` (154), `type_error:java` (138), `value_error:list_dict_count` (35) |
| Relevance checks | `irrelevance_error:decoder_success` (5,925), `relevance_error:decoder_failed` (1,386) |
| Multi-turn failures | `multi_turn:empty_turn_model_response` (20,553, "Model response list is empty for turn N"), `multi_turn:instance_state_mismatch` (11,271), `multi_turn:execution_response_mismatch` (7,153), `multi_turn:force_terminated` (71) |
| Server/harness (Harmony protocol) | `server:harmony_error_parser` (9,010), `server:bad_input_type` (1,066), `server:probable_doom_loop` (606), `server:harmony_error_header` (262), `server:harmony_error_role` (2), `server:inference_error` (2), `server:fatal_type_hallucination` (2), `server:bad_output_type` (1), `server:harmony_error_token_sequence` (1) |

**The irrelevance paradox, visible in the data.** The BFCL irrelevance scorer marks a run correct whenever it did not call one of the relevant tools, and a server error string decodes as "no tool call". So 267 infrastructure-error runs on `irrelevance` are scored `correct=1`, while the only irrelevance failures are the 5,925 `decoder_success` rows where the model emitted a valid call it should not have ("Valid syntax. Successfully decode AST when it should not."). The high irrelevance accuracy (85 to 99% by configuration) therefore overstates healthy behavior; this is finding 1 of the report, and the reason the burrito backends show *lower* irrelevance scores than the vanilla ones: the harness forces a real answer instead of passing silently.

## 7. Raw data on disk

Every row in the CSV maps to a directory on disk named after its `run_name`; the CSV is the aggregation of what is in these directories, produced by `eval_aggregator.py`.

```
data/
  eval_results.csv            <- this dataset
  bfcl_v4/                    321 directories
    <run_name>/
      result/gpt-oss-20b-{chat,responses}/
        non_live/BFCL_v4_{simple_python,simple_java,simple_javascript,irrelevance}_result.json
        live/BFCL_v4_{live_simple,live_relevance}_result.json
        multi_turn/BFCL_v4_multi_turn_base_result.json
      score/data_{overall,non_live,live,multi_turn,agentic,format_sensitivity}.csv
    __discard_wrong_reasoning_effort_default_medium_instead_of_low_but_identical_results_ok/
  gpt_oss/                    122 directories
    <run_name>/<test>_openai__gpt-oss-20b-<effort>_temp1.0_<YYYYMMDD_HHMMSS>.json            (benchmark summary: score, std, chars)
    <run_name>/<test>_openai__gpt-oss-20b-<effort>_temp1.0_<YYYYMMDD_HHMMSS>_allresults.json (score, metrics, htmls, convos, metadata)
    <run_name>/<test>_openai__gpt-oss-20b-<effort>_temp1.0_<YYYYMMDD_HHMMSS>.html            (rendered report)
    __discard_browser_hits_botwalls_yay_knowledge_capitalism/
    __discard_httpx_not_playwright_browser/
```

- **BFCL result files** are line-delimited JSON: one line per task instance in that run, each with the task `id`, the parsed model output per turn, per-turn input/output token counts, per-turn latency, and `inference_log`: the full per-turn trace including instance state and every message, with the complete reasoning text.
- **BFCL score files** are leaderboard-style summaries for that one configuration and seed (rank, model, overall and per-category accuracy).
- **GPT-OSS files** come from OpenAI's evaluation suite: the small `.json` is the per-benchmark summary, `_allresults.json` the full detail (the `convos` field carries the complete model responses), and the `.html` the rendered report. File names encode benchmark, model, effort, temperature, and the run timestamp.
- **Configuration directory names** mirror `run_name`: `<backend>_<wire>_<effort>_t-1_b-1_f-<fc>_s-<seed>` for BFCL and `<backend>_<wire>_<effort>_t-1_b-1_be-<b>_pe-<p>_s-<seed>` for GPT-OSS. BFCL holds 20 (backend, wire, effort) families x 2 fc modes x 8 seeds = 320 run directories; GPT-OSS holds 15 families x 8 seeds = 120 (the burrito families split by python on/off, the three vanilla ones at medium only).
- **The `__discard_*` directories** hold runs excluded from the CSV and from every reported number, kept for transparency: the GPT-OSS browser runs killed when knowledge sites hit paywalls and bot walls, an earlier browser batch that used httpx instead of the playwright engine, and a BFCL batch whose effort was misconfigured as medium when it should have been low (its results proved identical to the correct batch).
- **`artifacts/`** holds the two jinja chat templates exactly as served (`chat_template.jinja` default, `chat_template_fixed.jinja` the fix) and `eval_specs.xlsx`, the specification of the run campaign: two sheets (`bfcl_v4`, `gpt_oss`), each opening with the server configuration (API key, model root, backend URLs, project root) followed by the per-run spec rows.
- **`../eval_versions.txt`** pins what was run: llama.cpp commit `3a479c913`, vLLM `v0.21.0`, and the model's date strings frozen at 2026-05-25 (`FAKETIME` and `SYSTEM_MESSAGE_DATE_CONFIG`), so date-dependent prompts were identical across all runs. The backends are May 2026 releases evaluated against the original August 2025 model weights.
- **`../scripts/`** holds the exact serve configuration for each backend (llama.cpp default vs fixed jinja, vLLM default vs fixed vs production) and `nvidia-power.service`, the 3090 power-management service used during the ~1,062 GPU-hour campaign.

## 8. Caveats for analysis

1. **The design is not a full factorial.** Phase 1 is the only matched cross-product. Concretely: the three vanilla backends exist only at medium effort (both wires on BFCL, responses-only on GPT-OSS); burrito ran the chat wire at medium only and low/high on responses; the preserved-thinking backends exist only on `multi_turn_base`; GPQA ran `fc_model=0` only; AIME25's `fc_model=1` rows coincide exactly with the 1,440 python-enabled runs; and `browser_enabled` is 0 everywhere. Filtering to a combination that was never run yields an empty frame, and "all backends" in a phase means the backends that phase ran.
2. **The browser tool is untested.** `browser_enabled` is 0 on every row and `n_tool_calls_browser` is 0 everywhere. The browser campaign was killed mid-run: the sites the model kept opening (wikipedia included) were paywalled or bot-walled, and SearXNG search results were not deterministic, and the whole point was reproducible data. The tool itself works in burrito; the dataset simply does not exercise it.
3. **The thinking/answer split is sparse.** Only responses-wire GPT-OSS rows carry `reasoning_token_count`/`response_token_count` (14,124 and 17,711 rows); vanilla llama.cpp records zero thinking tokens, so no effort-based thinking comparison can use it. `output_token_count` is the number to use for total spend.
4. **`is_error` is about infrastructure.** It is 1 only when the backend or harness failed to serve the request (plus the 71 force-terminated runs); wrong model answers have `is_error=0`. Use `correct` for accuracy, `failure_mode` for the taxonomy, and `error_type` for the cause of diagnosable failures.
5. **Multi-turn turns are 0-indexed**, and the `*_success` token and latency columns count successful turns only, so they are not the run totals on partially failed runs. A run with `mt_num_turns_total=0` never started (all 3,800 are infrastructure errors).
6. **Determinism measures shaped the data.** Date strings frozen at 2026-05-25; vanilla servers ran with prompt caching disabled and other settings gimped, which cut throughput to ~100 tokens/s (normally 200+) to keep runs deterministic; batch size 1; temperature 1.0. Throughput numbers in this dataset are therefore not the backends' best-case.
7. **The snapshot is dated.** Backends are May 2026 releases run against the original August 2025 weights, and the backends (especially the `fc_model=1` path) have improved since. The dataset records that point in time, not the current one.
8. **`model_name` is a mirror, not an axis.** It tracks `fc_model` (section 3); never use it to split data.

## 9. Where to look next

| Want | Look at |
|---|---|
| The findings, in full | [`report.md`](report.md) (seven findings) and its per-section copies in [`sections/`](sections/) |
| The reasoning-effort story in depth | [`story.md`](story.md), built on phase 6 cuts of this file |
| Every figure and its scope | [`plots.md`](plots.md) (the canonical record of all 44 figures, organized by phase) and the PNGs in [`../plots/`](../plots/) |
| The public write-up | [`post.md`](post.md) |
| Regenerate any figure | [`../eval_report.ipynb`](../eval_report.ipynb) (cells 1 to 11, one per phase) and [`../eval_helpers.py`](../eval_helpers.py) (`EvalPlotter`) |
| How this file was produced | [`../eval_aggregator.py`](../eval_aggregator.py) (scoring and aggregation, per suite: `BFCLAggregator`, `GPTOSSAggregator`) |
| The statistics behind the key findings | [`scripts/explore_data.py`](scripts/explore_data.py) |
| The exact serve setup for re-runs | [`../scripts/`](../scripts/) and [`../artifacts/eval_specs.xlsx`](../artifacts/eval_specs.xlsx) |
