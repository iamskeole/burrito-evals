# Burrito Report Plan

## RECOVERY NOTES - Written 2026-08-13 after context compaction

### WHERE WE ARE

**report.md is DONE and APPROVED.** The user confirmed "good job, i think we're good with report.md as the academic version."

**NEXT: Create `story.md`** - a standalone deep-dive focused EXCLUSIVELY on the "not all reasoning is created equal" story. This is the single biggest, most transferable finding from the entire 320K-row evaluation. The user wants this in excruciating detail with charts, tables, world-class data analysis and copywriting.

### WHAT STORY.MD IS

Not a summary of report.md. A standalone narrative that treats the reasoning-effort story as THE story, with all the data depth, charts, and analysis it deserves. Think: a long-form technical essay that could stand alone as a research note.

### THE CORE INSIGHT (THE HOOK)

The common assumption about reasoning effort is that it controls token budget: more effort = more thinking tokens = better answers. The data REJECTS this. Two distinct but related phenomena:

**PHENOMENON 1: At matched reasoning length, higher effort still wins dramatically.**

Bin the data by reasoning-token count, compare accuracy within each bin across effort levels:

- AIME25 at ~1,448 reasoning tokens (pooled backends, fc0): low=26.0%, medium=97.2%, high=100.0%
- AIME25 at ~2,896 tokens: low=16.5-21.2%, medium=88.4-93.0%, high=100%
- AIME25 at ~5,793 tokens: low=16.7-24.4%, medium=80.6-87.9%, high=100%
- GPQA at ~362 tokens: low=55.7%, medium=87.7%, high=100%
- GPQA at ~724 tokens: low=44-46%, medium=83.8-86.4%, high=92.5-98.4%
- GPQA at ~1,448 tokens: low=43.7-50.1%, medium=76.6-77.7%, high=90.6-95.8%

71 percentage points between low and medium at the same token budget on AIME25. This is NOT more tokens buying better answers. This is the effort setting changing what the model does with those tokens.

**PHENOMENON 2: Brute-forcing reasoning tokens degrades accuracy within each effort level.**

Each effort level has an optimal zone. Push past it and accuracy crashes:

AIME25 by reasoning token bin within effort:
- Low: 83.3% at ~181 tokens, crashes to 18.8% at ~2,896 tokens
- Medium: 100% at ~362-724 tokens, holds to ~5,793 tokens, then degrades to 30.2% at ~46,341
- High: 100% from ~1,448 to ~5,793 tokens, holds above 90% to ~46,341, then drops to 43.7%

GPQA by reasoning token bin within effort:
- Low: 72.9% at ~23 tokens, crashes to 0% at ~2,896 tokens
- Medium: 100% at ~45 tokens, degrades to 37.2% at ~23,170 tokens
- High: 100% at ~181-362 tokens, holds above 76% to ~23,170, drops to 0% at ~185,364

The model wanders and degrades when forced to think longer than its effort level supports.

### STANDARD EFFORT LEVEL COMPARISONS (for reference tables)

AIME25, burrito@llamacpp, fc0:
- Low: 38.3% accuracy, 1,954 output tokens, 1,362 reasoning tokens
- Medium: 73.8%, 7,444 output, 6,694 reasoning
- High: 83.8%, 30,196 output, 29,574 reasoning

AIME25, burrito@vllm, fc0:
- Low: 35.4%, 2,005 output, 1,468 reasoning
- Medium: 73.8%, 7,484 output, 6,614 reasoning
- High: 87.5%, 31,443 output, 30,702 reasoning

GPQA, burrito@llamacpp, fc0:
- Low: 55.4%, 288 output, 251 reasoning
- Medium: 67.5%, 1,968 output, 1,946 reasoning
- High: 71.9%, 17,188 output, 16,938 reasoning

GPQA, burrito@vllm, fc0:
- Low: 56.3%, 298 output, 267 reasoning
- Medium: 67.2%, 2,050 output, 2,009 reasoning
- High: 73.0%, 16,886 output, 16,784 reasoning

Key: llama and vllm show nearly identical curves. Cross-backend consistency is a feature of the story.

### CHARTS TO USE (all in burrito-evals/plots/)

**Matched reasoning length (THE KEY CHARTS):**
- `phase_6-f15-reasoning_effort_matched_tokens-pooled.png` - Accuracy at matched reasoning length, pooled runs. Shows low (circles) consistently below medium (diamonds) below high (squares) at every overlapping token bin. Two panels: AIME25 (left) and GPQA (right).
- `phase_6-f16-reasoning_effort_matched_tokens-within_question.png` - Same but within-question averaging. Confirms effect is not driven by subset of easy/hard problems.

**Effort bin degradation (CRUCIAL SECOND STORY):**
- `phase_6-f01-reasoning_effort_bins-non_bfcl.png` - Correct progression by reasoning_token_count. 3 columns (Low/Medium/High), 2 rows (AIME25/GPQA). Shows each effort level peaking then crashing as tokens increase. Shaded bands = variance across 8 seeds.
- `phase_6-f02-reasoning_effort_bins-multiturn.png` - Same pattern on multi-turn by output_token_count.
- `phase_6-f03-reasoning_effort_bins-all_bfcl.png` - Same pattern across all BFCL tests.

**Effect-cost-tradeoff:**
- `phase_6-f04-reasoning_effort_story-bfcl_pooled.png` - Three panels: Effect (accuracy vs effort), Cost (median tokens vs effort), Tradeoff (accuracy vs tokens with Pareto front). BFCL pooled.
- `phase_6-f08-reasoning_effort_story-non_bfcl.png` - Same three panels for AIME25+GPQA pooled.

**Per-test effort stories:**
- `phase_6-f09-reasoning_effort_story-aime25.png` - AIME25 effort story
- `phase_6-f10-reasoning_effort_story-gpqa.png` - GPQA effort story
- `phase_6-f05-f07` - BFCL per-test effort stories
- `phase_6-f11-f14` - Per-test effort stories

### NARRATIVE ARC FOR STORY.MD

1. **Opening hook** - The assumption that reasoning effort = more tokens = better answers. Why everyone believes this. Why it is wrong.
2. **The matched-length revelation** - Same tokens, different effort, dramatically different accuracy. The 26% vs 97% number. This is the money shot.
3. **The degradation curve** - More tokens within an effort level degrades accuracy. Each effort has an optimal zone. The model wanders past it.
4. **Cross-backend consistency** - Same curves on llama.cpp and vLLM. Same across AIME25, GPQA, multi-turn, BFCL. This is a property of the model's reasoning architecture, not an artifact.
5. **The tradeoff landscape** - Effect vs cost vs tradeoff. Where medium sits on the Pareto front.
6. **Transferable lessons** - Any model with configurable reasoning depth will show this. Test empirically. More is not always better.

### TONE AND STYLE

- NO AI slop: no "however," "nevertheless," "not X but Y," em dashes, inverse/surprise reveals
- Human(e) tone: direct, clear, like a technical blog post or Substack essay
- World-class copywriting: this is THE story, make it scream
- Data-driven: every claim backed by numbers from the dataset
- No hedging: state findings clearly
- Accessible but rigorous

### DATA SOURCE

- CSV: `/home/p/code/local/burrito-evals/data/eval_results.csv` (320K rows)
- Helper methods: `/home/p/code/local/burrito-evals/eval_helpers.py`
  - `plot_effort_at_matched_tokens()` - generates the matched-length charts (f15, f16)
  - `plot_effort_story()` - generates the effect-cost-tradeoff charts (f04, f08)
  - Both return `(fig, axes, data)` so you can access the underlying data
- All plots in `/home/p/code/local/burrito-evals/plots/`
- Reference existing `report.md` for prose style and data accuracy

### KEY NUMBERS TO VERIFY (from our Python queries)

Matched reasoning length, AIME25, fc0, pooled backends:
- ~1,448 tokens: low=26.0%, medium=97.2%, high=100.0%
- ~2,896 tokens: low=16.5-21.2%, medium=88.4-93.0%, high=100%
- ~5,793 tokens: low=16.7-24.4%, medium=80.6-87.9%, high=100%

Matched reasoning length, GPQA, fc0, pooled backends:
- ~362 tokens: low=55.7%, medium=87.7%, high=100%
- ~724 tokens: low=44-46%, medium=83.8-86.4%, high=92.5-98.4%
- ~1,448 tokens: low=43.7-50.1%, medium=76.6-77.7%, high=90.6-95.8%

Degradation within effort, AIME25, pooled:
- Low: 83.3% -> 78.1% -> 61.6% -> 26.0% -> 18.8% -> 20.5%
- Medium: 100% -> 100% -> 97.2% -> 90.7% -> 84.3% -> 63.3% -> 47.2% -> 30.2% -> 0%
- High: 100% -> 100% -> 100% -> 96.9% -> 94.4% -> 90.6% -> 43.7%

### WHAT TO DO NEXT

1. ~~Write `story.md` as a single self-contained markdown file~~ **DONE**
2. ~~Focus ONLY on the reasoning story~~ **DONE** - no irrelevance paradox, no jinja fix, no preserved thinking
3. ~~Use ALL the charts~~ **DONE** - 9 charts referenced (f01, f02, f03, f04, f08, f09, f10, f15, f16)
4. ~~Include tables with the raw numbers~~ **DONE** - 12+ tables with matched-length, degradation, and cross-benchmark data
5. ~~Make the narrative compelling and data-rich~~ **DONE**
6. ~~Reference plots with `../burrito-evals/plots/<filename.png>`~~ **DONE**

### STORY.MD STATUS: COMPLETE

Written 2026-08-13. 18KB markdown. Narrative arc:
- The Assumption (hook)
- The Experiment (setup)
- The Matched-Length Revelation (money shot: 26% vs 97% at same token budget)
- The Degradation Curve (brute-forcing crashes accuracy)
- Cross-Benchmark Consistency (llama.cpp = vLLM, same curves)
- The Tradeoff Landscape (effect/cost/tradeoff, Pareto front)
- Transferable Lessons (two lessons + test empirically)
- Methodology

All data triple-checked against fresh Python queries. All plot paths verified.

### SESSION CONTEXT

The user provided the full original request (see below in this PLAN.md) which includes:
- The backstory of gpt-oss-20b release and community reception
- The burrito harness creation story
- The 4-phase evaluation design
- The 4 transferable lessons the user identified
- Style requirements (no AI slop, human tone)
- The user ran this on a single RTX 3090, batch size 1, temperature 1.0, 8 seeds
- 320K rows, 300K+ questions, 900 hours, 2 billion tokens
- User drove physical burritos around town while running evals (hence the name)

### WHAT WE CHANGED IN report.md (for reference)

Finding 6 was completely rewritten to focus on:
1. Matched reasoning length data (26% vs 97% at same token budget)
2. Brute-forcing degradation (each effort level has optimal zone)
3. Effect-cost-tradeoff with Pareto front
4. Cross-backend consistency
5. Transferable lessons

Removed: "schema tools beat more reasoning" subsection (belongs in Finding 4).
Updated: Executive summary and conclusion to lead with matched-length data.

---

## Resume Instructions (for next session - ORIGINAL)

This workspace contains a completed Phase 0 (data exploration) and a plan to write a single self-contained Markdown report about the burrito evaluation project. The data is in `burrito-evals/data/eval_results.csv` (320K rows), plots are in `burrito-evals/plots/`, and analysis helpers are in `burrito-evals/eval_helpers.py`. The exploration script at `scripts/explore_data.py` has been run and its findings are captured in `notes.md`.

**How to continue:**
1. Read this PLAN.md to understand current state and which phases are done.
2. Read `notes.md` for the raw data findings that feed each section.
3. The user will prompt which phase to work on next. Each phase writes one section file into `sections/`.
4. After all sections are written, Phase 12 assembles them into `report.md`.
5. Update this PLAN.md as you go -- check off completed phases, note any pivots.
6. Prose guidelines: no AI slop (no "however," "nevertheless," "not X but Y," em dashes, inverse/surprise reveals). Human(e) tone, direct and clear, like a technical blog post.
7. All figure references should point to `../burrito-evals/plots/<filename.png>` relative to the final report location, or the report can be placed at the repo root alongside `burrito-evals/`.
8. If context compacts, re-read this file and `notes.md` to recover state.

## Goal
Produce a single, self-contained Markdown report (`report.md`) that tells the data-driven story of the burrito evaluation project. Focus on transferable lessons for the broader community.

## Status
- [x] Phase 0: Data exploration scripts and key findings extraction
- [x] Phase 1: Write report section -- Executive Summary
- [x] Phase 2: Write report section -- Background (model, problem, burrito)
- [x] Phase 3: Write report section -- Experimental Setup
- [x] Phase 4: Write report section -- Finding 1: The Irrelevance Paradox
- [x] Phase 5: Write report section -- Finding 2: The Jinja Fix
- [x] Phase 6: Write report section -- Finding 3: Wire API Differences
- [x] Phase 7: Write report section -- Finding 4: Multi-Turn Base (Agentic Work)
- [x] Phase 8: Write report section -- Finding 5: Preserved Thinking
- [x] Phase 9: Write report section -- Finding 6: Not All Reasoning is Created Equal
- [x] Phase 10: Write report section -- Finding 7: Python Tool Impact
- [x] Phase 11: Write report section -- Conclusion / Lessons
- [x] Phase 12: Final assembly, cross-reference figures, polish prose

## Workspace Layout
```
workspace/
  PLAN.md              <- this file
  notes.md             <- scratchpad with key data findings
  report.md            <- final deliverable
  sections/            <- individual section drafts
  scripts/
    explore_data.py    <- data exploration (run complete, results in notes.md)
```

## Key Data Findings (confirmed from 320K row dataset)

### Finding 1: The Irrelevance Paradox
- BFCL irrelevance test marks ANY non-tool-call as correct, including silent failures
- 5,925 "model" failures scored as correct because no tool was called
- Burrito backends score LOWER on irrelevance (87%) vs vanilla (93%) because burrito actually tries to answer instead of silently passing
- Plot: phase_1-f01 (irrelevance), phase_1-f05 (error rates)

### Finding 2: The Jinja Fix (one-word change, massive impact)
Default jinja on chat API, fc_model=0:
- live_relevance: 3.1% -> 41.4% (+38.3%)
- live_simple: 3.1% -> 36.7% (+33.7%)
- simple_java: 18.0% -> 60.5% (+42.5%)
- simple_javascript: 13.0% -> 56.5% (+43.5%)
- simple_python: 1.8% -> 36.6% (+34.8%)
- multi_turn_base: 0.2% -> 14.6% (+14.3%)
- irrelevance: 99.9% -> 93.8% (-6.1%) -- goes DOWN because model actually tries
- Fix: remove "commentary" from valid channels when no tools present
- Plot: phase_1-f01 (compare default-jinja vs fixed-jinja bars)

### Finding 3: Wire API Differences
- vLLM vanilla responses API: multi_turn fc1 = 36.1% accuracy with 28.6% error rate (vs 23.9% / 4.9% on chat)
- vLLM responses API adds massive error rates on multi-turn
- Burrito sidesteps this by always using /completions endpoint
- Plot: phase_1-f01 (chat vs responses encoding), phase_1-f05 (error rates)

### Finding 4: Multi-Turn Base (Agentic Work)
- fc_model=1 (schema tools) dramatically beats fc_model=0 (AST parsing)
- burrito@llamacpp fc1: 52.2% vs fc0: 17.2%
- 68% of failures happen at turn 0
- Vanilla backends with fc0: 0-0.2% (essentially broken)
- Plot: phase_1-f01 (multi_turn_base section), phase_5-f07 (turn survival)

### Finding 5: Preserved Thinking is Not a Silver Bullet
- multi_turn fc=0: pt=15.2% vs standard=17.2% (slightly worse)
- multi_turn fc=1: pt=55.1% vs standard=51.8% (slightly better)
- Variance is lower for preserved thinking versions
- Net effect: marginal at best
- Plot: phase_5-f01, phase_5-f07

### Finding 6: Not All Reasoning is Created Equal (BIGGEST STORY)
AIME25 (burrito@llamacpp, fc=0): low=38.3% (1.9k tok) -> medium=73.8% (7.4k tok) -> high=83.8% (30.2k tok)
GPQA: low=55.4% (288 tok) -> medium=67.5% (1.9k tok) -> high=71.9% (17.2k tok)
multi_turn fc=0: low=10.5% -> medium=17.3% -> high=44.2%
- low->medium: +35.5% accuracy for 4x tokens (AIME25)
- medium->high: +10% for 4x tokens (diminishing returns)
- fc_model=1: more accurate AND cheaper (83.3% at 4.7k vs 73.8% at 7.4k)
- Plot: phase_6-f04 (BFCL pooled), phase_6-f09 (AIME25), phase_6-f10 (GPQA), phase_6-f15 (matched tokens)

### Finding 7: Python Tool Impact
- low effort AIME25: +21-24% accuracy with python tool
- medium effort: +10-12% improvement
- high effort: mixed (+2.5% llama, -5.8% vllm)
- Python calls reduce token usage at medium effort
- Plot: phase_4-f01

## Writing Guidelines
- No AI slop: no "however," "nevertheless," "not X but Y," em dashes, inverse/surprise reveals
- Human(e) tone: direct, clear, like a technical blog post
- Academic style but accessible
- Each section references specific plots from burrito-evals/plots/
- Focus on transferable lessons

## Process
1. Write each section as separate markdown file in sections/
2. Assemble into final report.md
3. Polish and verify all figure references
