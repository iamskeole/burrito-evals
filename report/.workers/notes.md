# Scratchpad Notes

## Key Data Findings (from explore_data.py run)

### Irrelevance Paradox
- Error rates are near zero (0.0-0.1%) across all backends
- 5,925 "model" failures in irrelevance -- these are cases where the model failed but the benchmark scored them correct because no tool was called
- The paradox: high accuracy numbers (87-93%) mask the fact that many are silent failures
- Burrito backends have LOWER irrelevance accuracy (87%) vs vanilla (93%) because burrito actually tries to answer instead of silently passing

### Jinja Fix -- MASSIVE Impact
Chat API, fc_model=0 (AST parsing, no tools in request):
- live_relevance: 3.1% -> 41.4% (+38.3%) -- default jinja was essentially broken
- live_simple: 3.1% -> 36.7% (+33.7%)
- simple_java: 18.0% -> 60.5% (+42.5%)
- simple_javascript: 13.0% -> 56.5% (+43.5%)
- simple_python: 1.8% -> 36.6% (+34.8%)
- multi_turn_base: 0.2% -> 14.6% (+14.3%)
- irrelevance: 99.9% -> 93.8% (-6.1%) -- goes DOWN because model actually tries to answer

The default jinja template was making the model hallucinate commentary channels and essentially break on all non-tool-call tasks. The fix is literally removing "commentary" from valid channels when no tools are present.

### Wire API Differences
vLLM vanilla on responses API:
- multi_turn_base fc=1: 36.1% accuracy with 28.6% error rate (vs 23.9% / 4.9% on chat)
- The responses API adds significant error rate on vLLM
- Burrito sidesteps this entirely by always using /completions endpoint

### Multi-Turn Base
fc_model=1 (schema tools) dramatically beats fc_model=0 (AST parsing):
- burrito@llamacpp fc1 chat: 52.2% vs fc0: 17.2%
- burrito@vllm fc1 chat: 56.2% vs fc0: 17.8%
- vanilla backends with fc0: 0-0.2% (essentially broken)
- 68% of failures happen at turn 0 (28,924 out of 42,848 failures)

### Preserved Thinking
On multi_turn_base (medium effort):
- fc=0: pt@llama=15.2% vs b@llama=17.2% (slightly WORSE with preserved thinking)
- fc=1: pt@llama=55.1% vs b@llama=51.8% (slightly BETTER)
- Std is lower for pt versions (less variance)
- Net effect: marginal at best, not a silver bullet

### Reasoning Effort (BIGGEST STORY)
AIME25 (burrito@llamacpp, fc=0):
- low: 38.3% (1,954 output tokens, 1,362 reasoning tokens)
- medium: 73.8% (7,444 output tokens, 6,694 reasoning tokens)
- high: 83.8% (30,196 output tokens, 29,574 reasoning tokens)

GPQA (burrito@llamacpp, fc=0):
- low: 55.4% (288 output tokens, 251 reasoning tokens)
- medium: 67.5% (1,968 output tokens, 1,946 reasoning tokens)
- high: 71.9% (17,188 output tokens, 16,938 reasoning tokens)

Key insight: For AIME25, low->medium gives +35.5% accuracy for 4x tokens. Medium->high gives +10% for 4x tokens. Diminishing returns but the model is doing fundamentally different things at each level.

fc_model=1 consistently uses fewer tokens (the model thinks less when it has structured tool definitions):
- AIME25 medium fc1: 73.8% -> 83.3% accuracy, 7,444 -> 4,678 tokens (more accurate AND cheaper)

### Python Tool Impact
AIME25 with python_enabled=1:
- low effort: +21-24% accuracy improvement (massive)
- medium effort: +10-12% improvement
- high effort: mixed (+2.5% on llama, -5.8% on vllm)
- Python calls also reduce token usage at medium effort (7,444 -> 4,678 tokens)

## Report Structure Notes
- The jinja fix story is the most surprising: a one-word change that fixes broken behavior
- The reasoning effort story is the most transferable: effort changes WHAT the model does, not just HOW MUCH
- The irrelevance paradox is the most important warning for the community
- Multi-turn + fc_model split shows that tool definition format matters enormously for agentic work
- Preserved thinking is the most nuanced: marginal gains, lower variance, not worth the complexity

## Prose Style Notes
- No "however," "nevertheless," "not X but Y," em dashes
- No inverse/surprise reveals
- Direct, clear, like a technical blog post
- Human(e) tone
