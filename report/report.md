# Evaluating gpt-oss-20b: What 320K Runs Tell Us About Local Inference

## Table of Contents

- [Executive Summary](#executive-summary)
- [Background](#background)
- [Experimental Setup](#experimental-setup)
- [Finding 1: The Irrelevance Paradox](#finding-1-the-irrelevance-paradox)
- [Finding 2: The Jinja Fix](#finding-2-the-jinja-fix)
- [Finding 3: Wire API Differences](#finding-3-wire-api-differences)
- [Finding 4: Multi-Turn Base (Agentic Work)](#finding-4-multi-turn-base-agentic-work)
- [Finding 5: Preserved Thinking](#finding-5-preserved-thinking)
- [Finding 6: Not All Reasoning is Created Equal](#finding-6-not-all-reasoning-is-created-equal)
- [Finding 7: Python Tool Impact](#finding-7-python-tool-impact)
- [Conclusion](#conclusion)

---

# Executive Summary

This report evaluates gpt-oss-20b across 320,192 runs using the Big Function Calling Leaderboard, AIME25, and GPQA. We compare burrito (a model-specific inference harness) against vanilla llama.cpp and vLLM backends, testing wire APIs, function calling modes, reasoning effort levels, and native tool support.

**Configuration matters more than model capability.** The same model produces results from 0% to ~87% depending on template, tool format, wire API, and effort level — 0% for the vanilla clients with AST parsing and the broken template on multi-turn work, ~87% on AIME25 at high effort. Seven findings emerge from the data:

1. **The irrelevance paradox.** BFCL's irrelevance test marks any non-tool-call as correct, including silent failures. Systems that try to help score lower than systems that do nothing.
2. **The jinja fix.** Removing "commentary" from valid output channels when no tools are present moves accuracy from 3% to 40% on live tests. The default template was breaking the model.
3. **Wire API differences.** On vLLM, the responses API errors on 73.5–83.5% of multi-turn runs versus 29.0–36.0% on chat — even though it is the more accurate of the two. Using completions with correct prompt construction avoids the problem.
4. **Multi-turn needs schemas.** fc_model=1 (structured tool schemas) triples accuracy over fc_model=0 (AST parsing) on agentic workflows. The vanilla AST-parsed backends collapse to 0.0–0.2% — a fixed jinja template rescues vanilla llama.cpp to 14.6%, but it still trails burrito.
5. **Preserved thinking is marginal.** Slightly better with schema tools, slightly worse with AST parsing. Lower variance but not enough accuracy gain to justify added complexity for most use cases.
6. **Not all reasoning is created equal.** At the same reasoning token budget, low effort gives 38% accuracy on AIME25 while medium gives 97%. The effort setting changes the quality of reasoning at any given token count, not just the token count itself. Brute-forcing more tokens within an effort level degrades accuracy as the model wanders.
7. **Python tools help most at low effort.** +21-24 points at low effort, +10-12 points at medium, mixed at high. Python calls also reduce token usage by offloading computation.

**The biggest story for other models:** Not all reasoning is created equal. For the same token budget, low, medium, and high effort produce entirely different answers, and the pattern holds across backends and benchmarks at very different accuracy levels. This is the most transferable lesson from this evaluation.

**Bottom line:** The model is not the bottleneck. The stack around it is. Correct templates, structured tool schemas, medium reasoning effort, and enabled python tools produce the best accuracy-to-cost ratio. Burrito addresses these configuration issues at the harness layer, with correctness gains that justify added latency for systems that need reliable tool calling and multi-turn workflows.

Every chart in this report is excerpted from the complete 44-figure record kept in plots.md next to this document, and figure numbers follow plots.md's phase scheme (leading number = phase, e.g. Fig. 5.7 is phase 5, figure 7); plots.md carries the per-figure statistics quoted here.

---

# Background

## The Model

This report evaluates gpt-oss-20b, OpenAI's 20-billion-parameter open weight model. It runs in MXFP4 quantization, which keeps memory footprint low enough for a single consumer GPU (RTX 3090) while preserving most of the model's capability. The model supports 131k context and was trained with native tool calling, interleaved reasoning channels, python code execution, and browser interaction.

What makes gpt-oss different from earlier open models is its channel-based architecture. The model does not just produce text. It alternates between thinking, calling tools, and providing final answers, and it can backtrack into reasoning mid-conversation. This design is powerful but demands an inference stack that respects those patterns instead of forcing them into a standard chat template.

## The Problem

Running gpt-oss locally exposed a gap between inference engines and model capability. The two dominant backends, llama.cpp and vLLM, excel at next-token prediction speed and correctness. They are not designed to handle gpt-oss's dynamic conversation structure.

llama.cpp supports `/v1/chat/completions` and `/v1/responses` and implements function calling through grammar constraints — decoding under a format grammar that forces each output to fit a declared tool-call structure. Tool calling works for user-defined tools but uses a hardcoded `functions.` prefix that does not match the model's training namespace. The model was trained on `python` and `browser.*` calls, not `functions.python`. Grammar constraints also bias model output, which can produce correctly named tool calls with hallucinated arguments. And there is no support for the model's native python or browser tools — with no route for the native tool schemas, they are effectively ignored by the vanilla client: the post's third broken thing, "the tool schemas were ignored."

vLLM handles tool calling more naturally on both chat and responses endpoints. It has basic python and browser support through a separate demo server. The responses API introduces high error rates on multi-turn tasks (73.5% on responses vs 29.0% on chat with schema tools, and 83.5% vs 36.0% with AST parsing). The browser tool defaults to commercial APIs, which defeats the purpose of running locally.

Both backends use jinja templates to render conversations. When the model's channel expectations do not match the template, generation fails silently or produces broken output. The default jinja template for gpt-oss includes a "commentary" channel in valid outputs even when no tools are present. This causes the model to hallucinate commentary channels and break on basic non-tool-call tasks.

## Burrito

Burrito is an inference harness that sits between client applications and inference backends. It accepts standard OpenAI and Anthropic API inputs (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) and handles the work of rendering conversations, managing tool calls, and recovering from hallucinations.

Behind the scenes, burrito sends `/v1/completions` requests to either llama.cpp or vLLM for raw token generation. It then processes the output, handles tool execution, and manages multi-turn state. This architecture gives burrito several advantages:

- **Correct conversation rendering.** Burrito renders conversations per the model's training specification, building prompts that match the channel structure the model was trained on rather than relying on jinja templates that can mismatch.
- **Hallucination recovery.** When the model produces malformed tool calls or wrong channels, burrito tells the model what went wrong and lets it self-correct. This beats grammar constraints because the model understands its own mistakes.
- **Native python and browser tools.** The model's `python` and `browser.*` calls execute inside the harness. Browser search runs on a local SearXNG instance. Browser open uses a custom Playwright engine. No third-party APIs, no fees.
- **Consistent wire protocol.** Burrito always uses `/v1/completions` for backend communication, sidestepping the error rate differences between chat and responses APIs.

The trade-off is latency. Burrito adds a processing layer on top of raw inference. The question this report answers is whether the correctness gains are worth the cost, and where burrito helps or hurts relative to running backends directly.

## Evaluation Framework

We evaluate using the Big Function Calling Leaderboard (BFCL) test suite plus two external benchmarks: AIME25 (mathematical problem solving) and GPQA (graduate-level science questions). Tests run across the seven backend configurations tabulated in Experimental Setup, two wire APIs (chat, responses), two function calling modes (fc_model=0 for AST parsing, fc_model=1 for schema-based tools), and three reasoning effort levels (low, medium, high). Each configuration runs with 8 random seeds for statistical reliability.

The full dataset contains 320,192 evaluation rows — one row per run, a run being one model attempt at one benchmark item under one configuration and one seed. Accuracy, error rates, token counts, and multi-turn survival are tracked for every run.

Two more pieces of vocabulary from plots.md, which the findings below lean on: the data is organized into numbered phases, one per sweep (a fixed choice of backend group, wire API, and effort scope), and within a phase a "unit" is a single benchmark item — a task, problem, or question, depending on the benchmark — on which runs are scored. A figure's columns are those units, arranged over the backend/tool/effort grid.

---

# Experimental Setup

## Benchmarks

We evaluate across nine tests spanning four categories:

**BFCL non-live tests** cover function calling and tool use in controlled settings:
- `simple_python`, `simple_java`, `simple_javascript` -- single-turn tool calls in specific languages
- `irrelevance` -- the model should produce no tool calls when none are relevant
- 176,960 rows total

**BFCL live tests** evaluate tool calling against real APIs:
- `live_simple` -- straightforward live API calls
- `live_relevance` -- the model must determine which live tools apply
- 61,376 rows total

**Multi-turn** tests agentic workflows requiring multiple sequential tool calls:
- `multi_turn_base` -- multi-step agentic tasks, tracked turn by turn (the survival analysis runs through turn 6)
- 64,000 rows total

**GPT-OSS native benchmarks** test reasoning-heavy tasks:
- `AIME25` -- mathematical problem solving (3,600 rows)
- `GPQA` -- graduate-level science questions (14,256 rows)

## Backends

Seven backend configurations are compared:

| Backend | Description |
|---------|-------------|
| `burrito@llamacpp` | Burrito harness with llama.cpp backend |
| `burrito@vllm` | Burrito harness with vLLM backend |
| `burrito-pt@llamacpp` | Burrito with preserved thinking, llama.cpp backend |
| `burrito-pt@vllm` | Burrito with preserved thinking, vLLM backend |
| `llamacpp@default-jinja` | Vanilla llama.cpp with default chat template |
| `llamacpp@fixed-jinja` | Vanilla llama.cpp with fixed chat template |
| `vllm` | Vanilla vLLM |

Not every finding draws on all seven: the phases behind the charts use different subsets of them, and the phase titles in plots.md state the exact set for each chart. The full sweep (phase 1) excludes the two preserved-thinking variants; the schema-tools BFCL effort figure (Fig. 6.6) covers the five non-preserved-thinking backends, while its multi-turn counterpart (Fig. 6.7) covers all seven; and the preserved-thinking phase (phase 5) is built around the burrito stacks.

On the wire: the wire API is the client-facing endpoint. For the vanilla backends it is also the backend traffic, and they are tested on both `/v1/chat/completions` and `/v1/responses`. Burrito always speaks `/v1/completions` to the backend; the client-facing API (chat or responses) is still a measured dimension, which is why burrito rows are counted under both wires in the tables below.

## Configuration Dimensions

Each test runs across multiple settings:

- **Wire API (client-facing):** `chat` or `responses` (101,120 chat rows, 219,072 responses rows)
- **Function calling mode:** `fc_model=0` (AST parsing) or `fc_model=1` (schema-based structured tools)
- **Reasoning effort:** `low`, `medium`, or `high` (controls the model's internal reasoning depth)
- **Python tool:** enabled or disabled (1,440 rows with python enabled, all on AIME25). Not a free axis: python-enabled runs are the burrito AIME25 runs with fc_model=1, so this dimension locks onto the tool-mode axis and the burrito backends rather than crossing them (plots.md phase 4, Fig. 6.13).

All runs use temperature 1.0, batch size 1, and 8 random seeds for statistical reliability. The full dataset totals 320,192 rows — 3.49B tokens over 1,062 GPU hours on a single RTX 3090. All figures cited below are excerpted from plots.md, the 44-figure compendium of the exercise, which is the authoritative record for the chart details quoted in this report.

## Metrics

- **Accuracy** (`correct`) -- binary correctness per test case
- **Error rate** (`is_error`) -- fraction of runs that produced errors (timeouts, malformed responses, etc.)
- **Token counts** -- input tokens, output tokens, and reasoning tokens
- **Multi-turn survival** -- number of turns completed before failure, and turn index of first failure
- **Tool call counts** -- number of python and browser tool calls made

---

# Finding 1: The Irrelevance Paradox

The BFCL irrelevance test measures whether a model correctly refrains from calling tools when none are relevant. The scoring rule is simple: any response that does not contain a tool call counts as correct.

This rule creates a blind spot. The test rewards inaction. A model that correctly decides no tools are needed scores the same as a model that crashes, produces empty output, or never tries. The test cannot tell the difference.

## The Data

Error rates on the irrelevance test stay low for the burrito backends (at most 0.42%) and the default template, while the fixed jinja template carries the outlier of phase 1's per-task error breakdown (16.7–99.3% of units across the tasks with fc_model=0). Accuracy scores span 85.4% to 99.9%: the vanilla default-template client at 99.9%, burrito at 92.1–92.4%, the fixed template at 93.8% — all with fc_model=0 — and a common 85.4–86.1% for every backend once fc_model=1 is enabled. The failure mode breakdown reveals the structure:

- 47,568 responses marked `success` (correct = 1)
- 267 infrastructure errors also scored correct (correct = 1) because they produced no tool call
- 5,925 model failures scored incorrect (correct = 0) because the model tried to call a tool when it should not have

The 267 infra errors scored as correct are the silent failures the test cannot detect. They are a small fraction (0.5% of total), but they illustrate the structural problem: the benchmark has no way to verify that a non-tool-call response came from correct judgment rather than a broken system.

The larger signal is in the incorrect responses. Burrito backends produce far more of them: 1,959 for burrito@llamacpp and 2,051 for burrito@vllm, compared to 558 for llamacpp@default-jinja and 563 for vLLM; the remaining 794 of the 5,925 sit in the fixed-jinja variant. Burrito tries harder. When the model attempts a response and produces a tool call on a question that requires none, the benchmark marks it wrong. Vanilla backends with the default jinja template often produce no output at all, which the benchmark reads as correct.

This is why burrito scores lower on irrelevance than the default-template vanilla clients (92.1–92.4% versus 99.9% with fc_model=0), and why enabling fc_model=1 pulls every backend to a common 85.4–86.1%. Lower irrelevance accuracy means the system is more active, not less capable.

![Mean accuracy on the irrelevance test by backend](../plots/phase_1-f01-mean_correct.png)
> Fig. 1.1: Mean accuracy on the irrelevance test by backend. Burrito backends score lower because they attempt tool calls more often, and incorrect tool calls are penalized.

![Error rates by task, across all seeds](../plots/phase_1-f05-is_error.png)
> Fig. 1.5: Error rates by task and backend. Burrito's irrelevance error rate is at most 0.42% — its accuracy gap against the vanilla clients there is wrong calls, not errors — while the fixed jinja template is the phase's outlier, erroring on 16.7–99.3% of units with fc_model=0 (peaking at 99.25% on simple_python, 98.0% on multi_turn_base, and 96.9% on live_simple).

## Implications

The irrelevance test conflates correct restraint with system failure. A high score can come from the model correctly deciding no tools are needed, or from the system producing no output at all. The benchmark cannot distinguish between the two.

This is a warning for anyone using BFCL or similar benchmarks. A test that marks non-tool-calls as correct rewards silence. Systems that actively engage with the task will score lower, even when their behavior is more useful in practice.

---

# Finding 2: The Jinja Fix

The default jinja chat template for gpt-oss includes "commentary" as a valid output channel even when no tools are present in the request. This one template mismatch breaks the model on nearly every non-tool-call task.

Removing "commentary" from valid channels when no tools are defined fixes the problem. The accuracy gains are dramatic. PR submitted to the official model repo on [HuggingFace](https://huggingface.co/openai/gpt-oss-20b/discussions/274/files).

## The Numbers

On the chat API with fc_model=0 (AST parsing, no tools in request):

| Test | Default Jinja | Fixed Jinja | Change |
|------|:------------:|:-----------:|:------:|
| live_relevance | 3.1% | 41.4% | +38.3% |
| live_simple | 3.1% | 36.7% | +33.7% |
| simple_java | 18.0% | 60.5% | +42.5% |
| simple_javascript | 13.0% | 56.5% | +43.5% |
| simple_python | 1.8% | 36.6% | +34.8% |
| multi_turn_base | 0.2% | 14.6% | +14.3% |
| irrelevance | 99.9% | 93.8% | -6.1% |

The default template produces near-zero accuracy on live and simple tests. The model hallucinates commentary channels and generates broken output. The fixed template restores functional behavior across the board.

The one metric that drops is irrelevance (99.9% to 93.8%). This connects directly to Finding 1. With the fixed template, the model actually tries to answer questions instead of producing silent failures. The irrelevance test marks those attempts as wrong, so the score goes down even though the system is doing more.

## Why This Happens

gpt-oss was trained to alternate between channels: thinking, tool calling, commentary, and final response. The default jinja template tells the model that commentary is always a valid output option. When no tools are present, the model still tries to use commentary channels, which breaks the expected response structure.

The fix is to remove "commentary" from the valid channel list when the request contains no tools. The model then produces clean responses in the expected format.

## Implications

Removing one channel from the chat template moves accuracy from 3% to 40% on live tests. This is not a minor tuning adjustment. The default template shipped with the inference stack was making the model non-functional for basic tasks.

This finding applies beyond gpt-oss. Any model trained with structured channel output is sensitive to the template it receives at inference time. Mismatches between training structure and inference template produce silent failures that look like model incompetence. If your model seems broken on simple tasks, check the template before blaming the weights.

---

# Finding 3: Wire API Differences

The wire API a client uses to talk to an inference backend matters more than most operators realize. On vLLM, the `/v1/responses` API introduces much higher error rates than `/v1/chat/completions` on multi-turn tasks.

## Chat vs Responses on vLLM

On the multi_turn_base test with fc_model=1 (schema tools):

| Wire API | Accuracy | Error Rate |
|----------|:--------:|:----------:|
| chat | 23.9% | 29.0% |
| responses | 36.1% | 73.5% |

The responses API gains accuracy but at the cost of a 2.5× increase in error rate; with AST parsing the gap is wider still (36.0% on chat versus 83.5% on responses). Nearly three-quarters of runs on the responses wire produce errors instead of results, while the chat API is more reliable but less accurate on structured tool calls.

![Accuracy across wire APIs and backends](../plots/phase_1-f01-mean_correct.png)
> Fig. 1.1 (same chart as in Finding 1, shown for the multi-turn region): Mean accuracy on multi_turn_base by backend, fc_model, and wire API. Vanilla vLLM is the only client whose chat and responses values diverge — most visibly with schema tools, 23.9% versus 36.1%.

![Error rates by backend and wire API](../plots/phase_1-f05-is_error.png)
> Fig. 1.5 (same chart as in Finding 1, shown for the multi-turn region): Error rates by task, across all seeds. On multi_turn_base the vanilla vLLM pair splits by wire — 36.0% (chat) versus 83.5% (responses) with AST parsing, and 29.0% versus 73.5% with schema tools — while the burrito stacks stay at 4.0–16.0% (fc0) and 0.5–2.0% (fc1).

## How Burrito Avoids This

Burrito sidesteps the chat vs responses tradeoff entirely. It accepts both APIs from clients but always communicates with the backend using `/v1/completions`. This keeps error rates low while burrito's own logic handles conversation rendering and tool call parsing.

The `/v1/completions` endpoint is simpler than chat or responses. It takes a prompt and returns tokens. Burrito builds the prompt correctly using the model's training structure, then parses the output to extract tool calls, reasoning, and final responses. This two-step approach avoids the encoding issues that affect the higher-level APIs.

## Implications

If you run vLLM directly and use the responses API for tool calling, expect high error rates on multi-turn tasks. The chat API is more stable but less capable with structured outputs. Neither is ideal.

The takeaway is that the simplest wire protocol often produces the most reliable results when paired with correct prompt construction on the harness side. Complex APIs add abstraction that can hide bugs in the backend's encoding logic.

---

# Finding 4: Multi-Turn Base (Agentic Work)

Multi-turn tasks require the model to make a sequence of tool calls, process results, and continue until the task completes. These tasks expose the full gap between function calling modes and backend capabilities.

## fc_model=1 Dominates fc_model=0

The difference between schema-based tools (fc_model=1) and AST parsing (fc_model=0) is enormous on multi-turn tasks. At medium effort:

| Backend | fc=0 | fc=1 |
|---------|:----:|:----:|
| burrito@llamacpp | 17.3% | 51.4% |
| burrito@vllm | 17.8% | 56.1% |
| llama.cpp, default jinja | 0.25% | 50.4% |
| llama.cpp, fixed jinja | 14.6% | 50.4% |
| vanilla vLLM | 0.0% | 23.9% (chat) / 36.1% (responses) |

With fc_model=0, the vanilla backends collapse: vLLM at 0.0% and default jinja at 0.25%, unable to parse tool calls reliably enough to sustain a conversation; the fixed template rescues vanilla llama.cpp to 14.6%, still well below burrito's 17–18%.

Switching to fc_model=1 triples accuracy on burrito@llamacpp (17.3% to 51.4%) and burrito@vllm (17.8% to 56.1%), and puts even the vanilla stacks in the 24–50% range. Schema-based tool definitions give the model clear structure for generating and parsing tool calls across turns.

## Failures Concentrate at Turn 0

Turn 0 is the wall of multi-turn work. Three measurements show why:

- **The vanilla AST-parsed backends die almost at once.** The conditional step-survival curves measure, of the runs that reach a turn, the fraction that clears it: vLLM clears turn 0 at just 0.6%, default jinja at 2.0%, and on fixed jinja only 639 of 1,600 trajectories (a trajectory is one run's path through the turns) even reach turn 0.
- **The per-turn failures compound end-to-end.** Schema-tool configurations keep 0.9–7.1% of all runs alive through turn 6, AST-parsed configurations fall to ~0 by turns 3–4, and the vanilla fc=0 backends are gone by turn 1.
- **Turn 0 drives the headline.** Because the model fails to make the first tool call correctly before the conversation can start, multi-turn accuracy is driven primarily by single-turn tool-calling quality, and improving turn-0 reliability has outsized impact on overall success.

![Multi-turn accuracy by backend and fc_model](../plots/phase_1-f01-mean_correct.png)
> Fig. 1.1 (same chart as in Findings 1 and 3, shown for the multi-turn region): Multi-turn base accuracy. fc_model=1 dramatically outperforms fc_model=0. The vanilla AST-parsed backends are effectively non-functional (the fixed template partially rescues llama.cpp).

![Turn survival aggregation](../plots/phase_5-f07-turn_survival-agg.png)
> Fig. 5.7: Conditional step survival, for turns with at least 10 trajectories. Schema-tool cohorts stay healthy through turn 6 (per-step pass rates of 50–85%); AST-parsed cohorts start much weaker and bleed at every turn.

![Cumulative turn survival](../plots/phase_5-f08-turn_survival-cum.png)
> Fig. 5.8: End-to-end path survival (the product of per-turn rates). Schema-tool configurations keep 0.9–7.1% of all runs alive through turn 6; AST-parsed configurations are gone by turns 3–4; the vanilla fc=0 backends by turn 1.

## Implications

Tool definition format is the largest single factor in multi-turn performance. Schema-based definitions (fc_model=1) give the model the structure it needs to sustain conversations. AST parsing (fc_model=0) is unreliable for multi-step workflows.

For operators building agentic systems, the lesson is clear: use structured tool schemas. The accuracy gain from fc_model=0 to fc_model=1 is larger than any backend choice.

---

# Finding 5: Preserved Thinking

OpenAI recommends pruning thinking history from the conversation unless the model is still actively issuing tool calls. The logic is that stale reasoning chains add noise without contributing to the next decision. Burrito follows this recommendation by default, stripping thinking tokens after each turn.

We tested the opposite approach with preserved thinking (pt) variants. These keep the model's internal reasoning visible in the output across all turns rather than pruning it. The hypothesis is that preserving reasoning chains helps the model maintain coherence and reduces variance.

## Mixed Results on Multi-Turn

On multi_turn_base at medium reasoning effort:

| Backend | fc=0 | fc=1 |
|---------|:----:|:----:|
| burrito-pt@llamacpp | 15.2% | 55.1% |
| burrito@llamacpp | 17.3% | 51.4% |

With fc_model=0, preserved thinking is slightly worse (15.2% vs 17.3%). With fc_model=1, preserved thinking is slightly better (55.1% vs 51.4%). The direction flips depending on tool calling mode.

Standard deviation across seeds is lower for preserved thinking with schema tools (fc=1): 1.5% for pt vs 2.7% for standard burrito. With AST parsing (fc=0), the pattern reverses slightly (3.0% for pt vs 2.8% for standard). The variance reduction is real but limited to the fc=1 setting.

![Preserved thinking accuracy on multi-turn](../plots/phase_5-f01-mean_correct.png)
> Fig. 5.1: Preserved thinking vs standard burrito on multi-turn base. Gains are marginal and direction depends on fc_model.

## Minimal Impact on BFCL Tests

On single-turn BFCL tests (non-multi-turn, medium effort), preserved thinking shows small deltas that flip in direction across different tests. The effect is not consistent enough to recommend pt as a general improvement.

## Implications

Preserved thinking is not a silver bullet. OpenAI's recommendation to prune thinking history unless the model is still issuing tool calls holds up under testing. Keeping all thinking history produces only marginal accuracy changes: slightly better with schema tools (fc=1, +3.7 points), slightly worse with AST parsing (fc=0, -2.1 points). The gains are small enough that we are not sure they are statistically significant.

Variance does decrease with preserved thinking on schema tools (1.5% std dev vs 2.7% for standard burrito), which is the one consistent positive signal. But the trade-off is complexity. Preserved thinking adds processing overhead and changes the output format. For most use cases, the accuracy gains do not justify the added complexity. Systems that use schema tools and need consistent performance across runs may find the lower variance worth it.

---

# Finding 6: Not All Reasoning is Created Equal

This is the biggest story from this evaluation and the finding most likely to transfer to other models.

The common assumption about reasoning effort is that it controls token budget: more effort means more thinking tokens, which means better answers. The data rejects this. Reasoning effort changes the quality of reasoning at any given token budget. Give the model the same question and the same number of reasoning tokens, and low, medium, and high effort still produce entirely different answers with dramatically different accuracy.

## At Matched Reasoning Length, Higher Effort Still Wins

The clearest evidence comes from matching reasoning token counts across effort levels. Bin the data by reasoning-token count, then compare accuracy within each bin.

**AIME25 at ~1,448 reasoning tokens** (pooled runs, burrito backends, both tool modes):

| Effort | Accuracy at ~1,448 reasoning tokens |
|--------|:-----------------------------------:|
| Low | 38.3% |
| Medium | 97.1% |
| High | 100.0% |

59 percentage points between low and medium at the same token budget. This is not more tokens buying better answers. This is the effort setting itself changing what the model does with those tokens.

**GPQA at ~362 reasoning tokens:**

| Effort | Accuracy at ~362 reasoning tokens |
|--------|:---------------------------------:|
| Low | 55.7% |
| Medium | 87.7% |
| High | 100.0% |

The gap persists across the full range of overlapping bins. At ~724 tokens on GPQA: low=45%, medium=85%, high=95%. At ~1,448 tokens: low=44%, medium=77%, high=93%. The hierarchy is stable. Higher effort always produces better accuracy at the same reasoning budget.

![Accuracy at matched reasoning length, pooled runs](../plots/phase_6-f15-reasoning_effort_matched_tokens-pooled.png)
> Fig. 6.15: Runs binned by reasoning-token budget (burrito backends, both tool modes, bins with at least 8 runs; 7 and 9 overlapping bins per test). Higher effort stays more accurate within essentially every shared bin — at ~1.4k reasoning tokens AIME25 sits at 38.3/97.1/100.0% for low/medium/high, and GPQA at ~724 tokens at 45.3/85.1/95.4% — and the shared right tail tells the overthinking story, with runs at ~90k+ reasoning tokens collapsing to ~38–44%.

![Accuracy at matched reasoning length, within question](../plots/phase_6-f16-reasoning_effort_matched_tokens-within_question.png)
> Fig. 6.16: Averaging each question first, so every AIME25 and GPQA question contributes equally within a token bin, leaves the ordering intact — at ~1.4k tokens, 39.1/97.0/100.0% on AIME25, and 44.4/85.6/96.9% on GPQA at ~724 tokens. The effect is not driven by which problems happen to land in a bin.

## Brute-Forcing Reasoning Tokens Degrades Accuracy

A second pattern emerges when we look within each effort level. Accuracy does not increase monotonically with reasoning token count. Each effort level has an optimal zone, and pushing past it degrades performance.

**AIME25 accuracy by reasoning token bin within each effort level:**

| Reasoning tokens | Low | Medium | High |
|:-----------------|:---:|:------:|:----:|
| ~181 | 100.0% | -- | -- |
| ~362 | 78.1% | 100.0% | -- |
| ~724 | 61.6% | 100.0% | -- |
| ~1,448 | 38.3% | 97.1% | 100.0% |
| ~2,896 | 18.8% | 90.7% | 100.0% |
| ~5,793 | 20.5% | 84.3% | 100.0% |
| ~11,585 | 0.0% | 63.3% | 96.9% |
| ~23,170 | -- | 47.2% | 94.4% |
| ~46,341 | -- | 30.2% | 90.6% |

Low effort peaks at 100% in the 128–256-token bin (where the model answers quickly and correctly) and then collapses to 0% by the 8k–16k bin, forced to think longer than its effort level supports. The model wanders and degrades.

Medium effort holds ~100% from 256 through ~1k tokens, is still 88–93% at 2k–4k, and then declines. High effort stays above ~88% all the way to the 32k–64k bin — but the pooled right tail tells the overthinking story, with runs at ~90k+ reasoning tokens collapsing to ~38–44%.

![Correct progression by reasoning token count](../plots/phase_6-f01-reasoning_effort_bins-non_bfcl.png)
> Fig. 6.1: Accuracy by reasoning token count within each effort level. Low effort (left column) peaks early and crashes. Medium (center) holds steady longer. High (right) is the most stable. The shaded bands show variance across 8 seeds.

The same degradation on GPQA: low effort goes from 73% at 23 tokens to 0% at 2,896 tokens. Medium effort declines from 100% to 37% across its range. High effort holds above 76% for much longer but still drops to 0% at extreme token counts.

![Correct progression by output token count](../plots/phase_6-f02-reasoning_effort_bins-multiturn.png)
> Fig. 6.2: Same pattern on multi-turn tasks by output token count. Each effort level has a peak zone and degrades outside it.

![Correct progression across all BFCL tests](../plots/phase_6-f03-reasoning_effort_bins-all_bfcl.png)
> Fig. 6.3: The peak-then-degrade pattern holds across all BFCL tests and effort levels, with one exception: the irrelevance test stays at or near 100% throughout (finding 1).

## The Effect-Cost-Tradeoff

The full picture across effort levels shows three phases:

1. **Effect.** On BFCL tasks, accuracy rises from low to medium effort: schema tools (fc_model=1) peak at medium (76.1–76.7%) and slip slightly at high (74.3–75.9%), while AST parsing (fc_model=0) keeps climbing all the way to high (~75–76%) at 16–17× the low-effort token cost. On reasoning-heavy tasks (AIME25, GPQA), accuracy keeps climbing from low through high, but with diminishing returns.
2. **Cost.** Median token count increases exponentially from low to medium to high.
3. **Tradeoff.** Medium effort with schema tools sits on the Pareto front: highest accuracy per token.

![Reasoning effort effect-cost-tradeoff, BFCL](../plots/phase_6-f04-reasoning_effort_story-bfcl_pooled.png)
> Fig. 6.4: BFCL_v4 pooled on the responses wire, in effect / cost / tradeoff panels. Schema tools already sit at the top of the effect curve at low effort (73.8–73.9%), peak at medium (76.1–76.7% at 377–407 median tokens), and high effort only reaches 74.3–75.9% while spending 1,362–1,975 tokens — medium with schema tools is the Pareto sweet spot.

![Reasoning effort effect-cost-tradeoff, AIME25+GPQA](../plots/phase_6-f08-reasoning_effort_story-non_bfcl.png)
> Fig. 6.8: AIME25 and GPQA pooled. The low→medium step carries the story: on burrito@llamacpp it is worth +15.2 points on fc_model=0 (53.1 → 68.3%) and +23.7 on fc_model=1 (59.6 → 83.3%), while medium→high adds just 5.2 and 2.9 more as median output tokens go from 606 to 31,924. (GPQA ran fc_model=0 only, so the fc_model=1 lines are AIME25 alone.)

## Implications

Two lessons transfer to any model with configurable reasoning depth.

**Effort changes answer quality, not just answer length.** At the same reasoning token budget, higher effort produces better accuracy because the model uses those tokens differently. Low effort at 1,448 tokens gives 38.3% on AIME25. Medium effort at the same budget gives 97%. These are not the same answers with different amounts of explanation. The effort setting changes the reasoning strategy, and different strategies produce different answers on the same questions.

**More tokens are not always better.** Within each effort level, accuracy peaks and then degrades as reasoning token count increases. Brute-forcing test-time compute by pushing the model to think longer than its effort level supports produces worse results. The model wanders, hallucinates, and loses track of the question. Each effort level has an optimal token zone.

The pattern is not an artifact of one backend or one benchmark: the same curves appear on llama.cpp and vLLM, and across AIME25, GPQA, multi-turn, and BFCL tests, at very different accuracy levels. The shape is a property of the model's reasoning architecture.

Test your effort levels empirically. Do not assume more reasoning is always better. Do not assume the model is just giving you the same answer with more words.

---

# Finding 7: Python Tool Impact

gpt-oss was trained with native python tool support. The model can write and execute code as part of its reasoning process. We test this capability on AIME25, where code execution should help with mathematical computation.

## Large Gains at Low Effort

With python enabled on AIME25:

| Effort | Accuracy Change |
|--------|:---------------:|
| Low | +21 to +24 points |
| Medium | +10 to +12 points |
| High | Mixed (+2.5 points llama, -5.8 points vllm) |

Python tools give the biggest boost at low reasoning effort, where the model does not think enough on its own to solve problems. At low effort, python execution acts as a computation engine that compensates for limited reasoning.

At medium effort, python still helps but less dramatically. The model is already doing substantial reasoning, so code execution fills gaps rather than carrying the load.

At high effort, results are mixed. The model generates enough reasoning to solve most problems without external computation. Python calls can help or hurt depending on whether the generated code is correct.

## Token Reduction

Python calls also reduce token usage at medium effort: with the python tool enabled, the model offloads computation to code execution instead of writing out full solutions, and at the same (medium) effort the python-enabled AIME25 runs think about half as much as the no-python runs (plots.md Fig. 6.13).

![Python tool impact on AIME25](../plots/phase_4-f01-mean_correct.png)
> Fig. 4.1: AIME25 accuracy with python tool enabled vs disabled. Largest gains at low reasoning effort.

## Implications

Python tool support is most valuable when reasoning effort is constrained. At low effort, it adds 21-24 points of accuracy. At medium effort, it adds 10-12 points while reducing token usage. At high effort, the benefit is mixed — the model already reasons its way to most solutions, so the toggle moves accuracy by a few points in either direction (+2.5 points on llama.cpp, −5.8 points on vLLM).

For production systems with latency or cost constraints, enabling python at medium effort gives a strong accuracy boost with lower token costs. The model uses code to verify calculations instead of generating long reasoning chains.

This finding reinforces the broader theme: the right tool configuration matters more than raw reasoning budget. Two views of the cost: at the same (medium) effort, the python-enabled AIME25 run thinks about half as much as the no-python run (the clean same-effort comparison, plots.md Fig. 6.13); and cross-effort, the python-enabled medium configuration on burrito@llamacpp (83.3% at 6,738 median output tokens) matches the no-python high-effort configuration (83.8% at 38,971 tokens) for about a fifth of the tokens. On burrito@vllm the accuracy pattern is less clear (85.8% vs 87.5%), but the token savings remain substantial.

---

# Conclusion

## Lessons for Operators

Seven findings from 320,192 evaluation rows. Here is what to do with them.

**Check your chat template before blaming the model.** The default jinja template for gpt-oss was breaking basic functionality. Removing "commentary" from valid channels when no tools are present moved accuracy from 3% to 40% on live tests. This applies to any model trained with structured output. Template mismatches produce silent failures that look like model incompetence.

**Use schema-based tool definitions.** fc_model=1 (structured schemas) triples accuracy over fc_model=0 (AST parsing) on multi-turn workflows and reduces token usage on reasoning tasks. On AIME25 at medium effort, schemas reach 83.3% accuracy at 6,738 tokens vs 73.8% at 11,816 tokens. Tool definition format is the largest single factor in agentic performance.

**Effort changes answer quality at any token budget.** At matched reasoning length, low effort gives 38.3% on AIME25 while medium gives 97%. The effort setting changes what the model does with its tokens, not how many it uses. Brute-forcing more tokens degrades accuracy as the model wanders past its optimal zone.

**Enable python tools at low and medium effort.** Python execution adds 21-24 points of accuracy at low effort and 10-12 points at medium effort, while reducing token usage. The model uses code to verify calculations instead of generating long reasoning chains.

**Beware benchmarks that reward silence.** The BFCL irrelevance test marks any non-tool-call as correct, including silent failures. Systems that actively engage with the task score lower than systems that produce no output. High accuracy on irrelevance does not mean the model correctly decided no tools were needed.

**Prefer simple wire protocols with correct prompt construction.** The `/v1/responses` API on vLLM produces 73.5–83.5% error rates on multi-turn tasks (29.0–36.0% on `/v1/chat/completions`), which is more stable but less accurate. `/v1/completions` with correct prompt building on the harness side avoids both problems.

## The Big Picture

Configuration choices matter more than raw model capability. The same gpt-oss-20b model produces results ranging from near-zero to ~87% accuracy depending on template, tool format, wire API, and effort level. The model is not the bottleneck. The stack around it is.

Burrito addresses this gap by handling conversation rendering, tool execution, and hallucination recovery at the harness layer. The trade-off is added latency over raw inference. The data shows the correctness gains are substantial for production systems that need reliable tool calling and multi-turn workflows.

Four lessons transfer beyond gpt-oss and burrito:

**Benchmarks can lie.** The irrelevance paradox shows that benchmarks can silently reward system failures or penalize systems that try harder. A high score does not always mean a capable model. Check what your benchmark actually measures.

**Dig into your infra.** A single channel in a jinja template moved accuracy from 3% to 40%. The default template shipped with the inference stack was making the model non-functional. Before scaling up, check the plumbing.

**Prune thinking history.** OpenAI's recommendation to strip thinking tokens unless the model is still issuing tool calls holds up. Preserved thinking adds marginal accuracy at best and increases complexity. The variance reduction is real but small.

**Not all reasoning is created equal.** This is the biggest story. At the same reasoning token budget, higher effort produces dramatically better accuracy because the effort setting changes the quality of reasoning, not just the quantity. At ~1,448 tokens on AIME25: low=38.3%, medium=97%, high=100%. Separately, brute-forcing more tokens within an effort level degrades accuracy as the model wanders past its optimal zone. Both patterns translate across backends and benchmarks. Any model with configurable reasoning depth will show similar behavior. Test empirically.

The broader lesson: any model trained with structured output patterns needs an inference stack that respects those patterns. Grammar constraints, jinja template mismatches, and API-level encoding bugs all fight the model instead of working with it. The best results come from stacks that render conversations per the training specification and let the model self-correct when it goes off track.
