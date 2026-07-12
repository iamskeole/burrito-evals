# Task brief: gpt-oss-20b backend/harness evaluation analysis

You have been given `eval_results_all.csv` — raw eval results for **gpt-oss-20b** (OpenAI's open-weights 20B model) tested across multiple inference backends and harnesses. Nothing else is provided; you do not have the source repo's README. This brief exists to give you the context that README would have provided, plus everything a first analysis pass got wrong before it got fixed. Read all of it before writing any code.

Your job: produce a rigorous, honest analysis report answering the research questions in section 4, in the style specified in section 6. Budget for this being a multi-hour, multi-step analysis — don't rush to a single pooled summary table.

## 1. What this dataset is

The CSV logs eval runs from a project comparing inference backends/harnesses for serving gpt-oss-20b, plus one custom harness ("burrito") built to fix tool-calling problems in the two mainstream backends. Evals are BFCL v4 (function/tool calling) and two of OpenAI's own published benchmarks (AIME25, GPQA).

**The backends/harnesses under test** (`backend` column, 5 values):
- `llamacpp@default-jinja` — vanilla llama.cpp, standard chat template.
- `llamacpp@fixed-jinja` — vanilla llama.cpp, but with a custom jinja template that specifically targets **non-function-calling (prompt-based) tool-call formatting**. It is not a general accuracy fix — verify this claim yourself in section 3 rather than taking it on faith.
- `vllm` — vanilla vLLM, provider-default serving.
- `burrito@llamacpp` — the custom "burrito" harness, using llama.cpp for next-token prediction on the raw `/completion` endpoint.
- `burrito@vllm` — same harness, using vLLM for next-token prediction instead.

**What "burrito" is, and why it exists:** llama.cpp and vLLM both have known tool-calling reliability problems. Burrito is a harness that sits *in front of* either backend (drop-in replacement for the chat completions endpoint), renders prompts/messages itself using the model's native Harmony format, talks to the backend at the raw completion level, and treats the model as an agent it can correct rather than a black box: it detects hallucinated tool names or doom-loops mid-generation and re-prompts the model to recover. It also is the only harness that (a) exposes gpt-oss's native sandboxed python + browser tools, and (b) allows reasoning effort to be set per-request rather than fixed at server startup.

**Other design axes in the data**, all present as columns — do not assume you know their semantics; confirm against `df.head()`, `.unique()`, and crosstabs first:
- `wire_api`: `chat` (OpenAI Chat Completions) vs `responses` (OpenAI Responses API).
- `fc_model`: whether the client passed an explicit tool schema (**FC**, `fc_model=1`) or the harness had to build the prompt itself and expect the model to emit tool calls in a specific format, parsed via AST (**non-FC**, `fc_model=0`). This is a BFCL-specific testing mode, not a universal property of every row — check whether it's even a meaningful axis for non-BFCL tests before using it.
- `reasoning_effort`: `low`/`medium`/`high`. Only meaningfully configurable per-request under burrito; other backends fix it at server startup.
- `browser_enabled` / `python_enabled` + `n_tool_calls_browser` / `n_tool_calls_python`: native tool access and usage counts. Confirm which backend(s) and which tests actually varied this before treating it as a general axis.
- `seed`: 8 fixed seeds, run at temperature 1.0 (the model's recommended default) specifically to probe seed-to-seed consistency, since real deployments won't pin a seed.
- `test_type` / `test_name` / `test_id`: benchmark category and individual sub-test. Expect roughly two families — a tool-calling suite with several sub-categories, and a reasoning/QA suite with a couple of named benchmarks.
- `input_token_count` / `output_token_count` / `latency` (seconds) / `is_error` (bool) / `correct` (bool, ground truth).

## 2. Do not assume symmetric experimental coverage

The backends were **not** all tested under the same conditions. Before writing a single comparison, build full crosstabs (`pd.crosstab`) of `backend × reasoning_effort`, `backend × wire_api`, `backend × fc_model`, `backend × test_type`, and `backend × browser_enabled`, and read them. Expect (but verify, don't assume) something like:

- Only the two burrito backends have `reasoning_effort` values other than `medium`.
- Only one burrito backend/reasoning-effort combination has `browser_enabled=1` / `python_enabled=1` rows, and likely only on the reasoning/QA suite, not the tool-calling suite.
- The reasoning/QA suite (AIME25/GPQA-style tests) probably only exists under one `wire_api` value and one `fc_model` value — there's no "chat vs responses" or "FC vs non-FC" axis to check there, because tool-calling mode is a meaningless concept for problems that don't call tools.
- The tool-calling suite (BFCL-style) probably has full `fc_model × wire_api` coverage for every backend.

**Any comparison across backends must first be restricted to a subset where the compared conditions were actually run identically** (same `reasoning_effort`, same tool-access setting). Define this "matched-conditions subset" explicitly, early, and reuse it — don't silently pool rows from different conditions into one backend-level number.

## 3. Confounds to explicitly check for (this is the part a first pass gets wrong)

Verify each of these empirically against the actual data — don't take this list on faith, and don't skip checking just because it's listed here. If the data contradicts something below, trust the data and say so.

1. **FC vs non-FC is probably the single largest driver of accuracy differences on the tool-calling suite, and pooling over it will hide that.** Split every tool-calling-suite comparison by `fc_model` before drawing conclusions. A backend that looks like it has a huge advantage "in aggregate" may have ~no advantage in FC mode and a massive one in non-FC mode (or vice versa). Report both, not just the pooled number.

2. **A template/config fix aimed at "non-FC formatting" should be checked for whether it does anything at all in FC mode.** If two backends are identical except for such a fix, compute their accuracy *separately by fc_model* — you may find they're statistically indistinguishable in FC mode (differences of exactly 0.0, p=1.0 are possible and meaningful, not a bug) and diverge sharply only in non-FC mode. If a fix improves non-FC accuracy, also check its **error rate** in non-FC mode specifically — a large accuracy gain can co-occur with a much higher outright failure rate than the pooled number suggests.

3. **Check `wire_api` (chat vs responses) per backend, restricted to the subset where both actually exist.** Do this only on the tool-calling suite if the reasoning/QA suite lacks wire_api coverage (see §2) — otherwise you'll compare "chat" (tool-calling suite only) against "responses" (tool-calling suite + reasoning/QA suite), which silently confounds wire-API effects with test-suite composition. Check both accuracy *and* error rate — a backend can show a small accuracy difference between wire APIs while having a much larger reliability (error-rate) gap that the accuracy number doesn't surface.

4. **Quantization/tensor-format comparisons are probably fully confounded with serving engine.** If one backend family always uses one quantization format and the other family always uses a different one, you cannot cleanly attribute an accuracy difference to quantization alone — say so explicitly rather than presenting it as a clean ablation.

5. **If you build any "task difficulty" proxy from a continuous column like input token count, check whether that column is itself correlated with `fc_model` (or another categorical axis) within each test category before using it.** A naive quartile-binned "difficulty" chart can just re-detect the FC/non-FC split instead of real difficulty, especially if one mode's prompts are systematically longer (e.g. because tool definitions get spelled out in natural language in one mode but not the other). Fix by stratifying the quartile computation within each combination of the confounding variable, not just within test category.

6. **Use the seed structure for more than a bigger sample size.** With 8 seeds, compute per-seed accuracy for each backend and look at the spread (std, min–max, coefficient of variation) in addition to pooled significance tests. Pooled two-proportion z-tests treat every row as an independent trial, which is optimistic given shared seeds; per-seed variance is the more honest robustness check and can surprise you — e.g. the most *accurate* configuration is not guaranteed to be the most *seed-stable* one, and that's worth flagging for anyone deciding whether to trust a single eval run.

7. **Check for rows where `is_error=1` and `correct=1` both hold.** Decide how to treat them (recommended default: trust `correct` as the grader's ground truth, keep the row as-is, and note the anomaly count in your caveats — don't silently recode).

8. **Do not invent a failure-mode taxonomy that isn't in the data.** If asked "which failure modes does X reduce" or similar, check whether there's any column beyond binary `is_error`/`correct` that could support that (e.g. an error-type or error-message field). If not, say plainly that the question can't be answered from this data without more granular logging, propose what instrumentation would be needed, and don't produce a proxy analysis that presents itself as a real answer to a question about mechanism.

9. **Latency is end-to-end wall-clock time.** Don't imply you know how it decomposes (queueing vs. inference vs. tool round-trips) unless there's a column that actually breaks it down.

10. **Sample sizes are probably very unequal across test categories.** If one benchmark family has far fewer rows per cell than another (especially once you start slicing by tool-enabled subsets), say so and treat those comparisons as lower-confidence, especially near conventional significance thresholds.

## 4. Research questions to answer

Organize the report around these. If a question turns out to be unanswerable for a reason like §3.8, state that plainly early (don't bury it at the end) and explain what data would be needed — don't force a partial or fabricated answer.

**Core performance**
1. Accuracy of each backend, across benchmarks and seeds — broken out by test category *and* by whichever axes in §3 turn out to matter (FC/non-FC, wire_api) rather than one pooled number.
2. Variance of performance across seeds.

**Harness/fix impact**
3. Effect size of the custom harness (and any other "fix" backend) on accuracy *and* error rate, split by whichever axis turns out to gate the effect (see §3.1–2).
4. Which failure modes does the harness reduce? — answer only if the data supports it (§3.8); otherwise state why not.

**Efficiency tradeoffs**
5. Latency and token cost *per correct answer* (not just per request) — a single efficiency number that penalizes wasted low-accuracy requests appropriately. Split by the same confounding axes if pooling would mislead.
6. Pareto frontier of accuracy vs. latency — check whether this needs to be split (e.g. by FC/non-FC) to avoid one backend's frontier position being an artifact of pooling two very different regimes together.

**Model behavior**
7. How does reasoning effort affect the accuracy/latency tradeoff? Check whether the effect is uniform or concentrated in a specific regime (e.g. does effort mostly help recover from formatting failures, or does it help uniformly regardless of mode?).
8. How does performance scale with task difficulty? Build a defensible proxy if no explicit difficulty label exists, explicitly checking it for the confound in §3.5 before presenting it.

**Tools**
9. On which tasks do native tools (if any are logged) improve performance, and by how much? Report usage rates (how often the tool was actually invoked when available), not just availability.
10. Do tools reduce specific failure modes? — same caveat as Q4.

**System design**
11. How robust is each backend to seed-driven stochasticity? (Builds on Q2 — frame this one for a production audience: what does seed variance imply for trusting a single benchmark run before shipping?)
12. Which backend is optimal under different production constraints? Produce a concrete decision table (scenario → recommendation → one-line justification tied to a specific number from earlier in the report), not just a verbal recommendation. Constraints worth considering: latency budget, error-rate tolerance, whether traffic is FC-only vs. mixed/non-FC, whether wire-API consistency matters, whether the workload is online vs. batch.

## 5. Statistical approach

- Two-proportion z-tests (or equivalent) for accuracy/error-rate comparisons within a matched-conditions subset; report exact or bucketed p-values, don't just say "significant."
- Complement pooled significance tests with the per-seed variance check from §3.6 wherever a claim rests on comparing two configurations' accuracy.
- When a proxy (like a difficulty binning) could be confounded, check it before using it, and say what you checked, not just present the final chart.

## 6. Deliverable format

- A single self-contained markdown report: store local images for charts, no external file dependencies, clean typography, a table of contents with anchor links, a short KPI/headline callout block near the top, a methodology-and-caveats section, and an appendix with the full per-test-name × backend breakdown (accuracy, error rate, latency) plus any per-axis split tables (e.g. FC vs non-FC) that are too granular for the main body.
- Tone: report what the data shows, including null results, reversals, and anything that complicates a clean narrative (e.g. a "fix" that helps accuracy but hurts reliability, or a harness that underperforms on one specific sub-test). Flag every confound you found and how you handled it. Don't round a nuanced, mode-dependent effect down to one aggregate number in the headline framing — if the honest answer is "it depends on X," lead with that.
- If the person only asked a narrow question, still do the full exploration/validation pass internally — it's what prevents the confounded-comparison mistakes in §3 — but scope the *written* deliverable to what they asked, and mention relevant confounds only where they'd change the answer.

## 7. Suggested workflow

1. Load the CSV; profile shape, dtypes, and every categorical column's unique values.
2. Build the crosstabs from §2; write down what you actually find (not what you expected).
3. Define the matched-conditions subset(s) you'll use for cross-backend comparisons; state the definition once, reuse it.
4. Compute core accuracy/error/latency tables sliced by test category and by every axis from §3 that turns out to matter — check §3 items 1–5 empirically before building any pooled chart.
5. Run the seed-variance check (§3.6).
6. Build charts that show the splits that matter — a pooled chart that hides a real effect is worse than no chart.
7. Write the report per §6, including a methodology/caveats section that states every subset definition, every confound checked, and every question you couldn't fully answer.
8. Sanity-check the output file before sharing it.