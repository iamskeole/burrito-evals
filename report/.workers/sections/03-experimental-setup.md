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
- `multi_turn_base` -- multi-step agentic tasks, 1-7 turns each, tracked turn by turn
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

Burrito backends use `/v1/completions` for all backend communication. Vanilla backends are tested on both `/v1/chat/completions` and `/v1/responses` wire APIs.

## Configuration Dimensions

Each test runs across multiple settings:

- **Wire API:** `chat` or `responses` (101,120 chat rows, 219,072 responses rows)
- **Function calling mode:** `fc_model=0` (AST parsing) or `fc_model=1` (schema-based structured tools)
- **Reasoning effort:** `low`, `medium`, or `high` (controls the model's internal reasoning depth)
- **Python tool:** enabled or disabled (1,440 rows with python enabled, all on AIME25)

All runs use temperature 1.0, batch size 1, and 8 random seeds for statistical reliability. The full dataset totals 320,192 rows.

## Metrics

- **Accuracy** (`correct`) -- binary correctness per test case
- **Error rate** (`is_error`) -- fraction of runs that produced errors (timeouts, malformed responses, etc.)
- **Token counts** -- input tokens, output tokens, and reasoning tokens
- **Multi-turn survival** -- number of turns completed before failure, and turn index of first failure
- **Tool call counts** -- number of python and browser tool calls made
