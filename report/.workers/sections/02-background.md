# Background

## The Model

This report evaluates gpt-oss-20b, OpenAI's 20-billion-parameter open weight model. It runs in MXFP4 quantization, which keeps memory footprint low enough for a single consumer GPU (RTX 3090) while preserving most of the model's capability. The model supports 128K context and was trained with native tool calling, interleaved reasoning channels, python code execution, and browser interaction.

What makes gpt-oss different from earlier open models is its channel-based architecture. The model does not just produce text. It alternates between thinking, calling tools, and providing final answers, and it can backtrack into reasoning mid-conversation. This design is powerful but demands an inference stack that respects those patterns instead of forcing them into a standard chat template.

## The Problem

Running gpt-oss locally exposed a gap between inference engines and model capability. The two dominant backends, llama.cpp and vLLM, excel at next-token prediction speed and correctness. They are not designed to handle gpt-oss's dynamic conversation structure.

llama.cpp supports `/v1/chat/completions` and `/v1/responses` and implements function calling through grammar constraints. Tool calling works for user-defined tools but uses a hardcoded `functions.` prefix that does not match the model's training namespace. The model was trained on `python` and `browser.*` calls, not `functions.python`. Grammar constraints also bias model output, which can produce correctly named tool calls with hallucinated arguments. There is no support for the model's native python or browser tools.

vLLM handles tool calling more naturally on both chat and responses endpoints. It has basic python and browser support through a separate demo server. The responses API introduces high error rates on multi-turn tasks (28.6% on multi_turn_base vs 4.9% on chat). The browser tool defaults to commercial APIs, which defeats the purpose of running locally.

Both backends use jinja templates to render conversations. When the model's channel expectations do not match the template, generation fails silently or produces broken output. The default jinja template for gpt-oss includes a "commentary" channel in valid outputs even when no tools are present. This causes the model to hallucinate commentary channels and break on basic non-tool-call tasks.

## Burrito

Burrito is an inference harness that sits between client applications and inference backends. It accepts standard OpenAI and Anthropic API inputs (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`) and handles the work of rendering conversations, managing tool calls, and recovering from hallucinations.

Behind the scenes, burrito sends `/v1/completions` requests to either llama.cpp or vLLM for raw token generation. It then processes the output, handles tool execution, and manages multi-turn state. This architecture gives burrito several advantages:

- **Correct conversation rendering.** Burrito renders conversations per the model's training specification, not through jinja conditionals. The model sees the channel structure it was trained on.
- **Hallucination recovery.** When the model produces malformed tool calls or wrong channels, burrito tells the model what went wrong and lets it self-correct. This beats grammar constraints because the model understands its own mistakes.
- **Native python and browser tools.** The model's `python` and `browser.*` calls execute inside the harness. Browser search runs on a local SearXNG instance. Browser open uses a custom Playwright engine. No third-party APIs, no fees.
- **Consistent wire protocol.** Burrito always uses `/v1/completions` for backend communication, sidestepping the error rate differences between chat and responses APIs.

The trade-off is latency. Burrito adds a processing layer on top of raw inference. The question this report answers is whether the correctness gains are worth the cost, and where burrito helps or hurts relative to running backends directly.

## Evaluation Framework

We evaluate using the Big Function Calling Leaderboard (BFCL) test suite plus two external benchmarks: AIME25 (mathematical problem solving) and GPQA (graduate-level science questions). Tests run across multiple backends (burrito with llama.cpp, burrito with vLLM, vanilla llama.cpp, vanilla vLLM), wire APIs (chat, responses), function calling modes (fc_model=0 for AST parsing, fc_model=1 for schema-based tools), and three reasoning effort levels (low, medium, high). Each configuration runs with 8 random seeds for statistical reliability.

The full dataset contains 320,000 evaluation rows. Accuracy, error rates, token counts, and multi-turn survival are tracked for every run.
