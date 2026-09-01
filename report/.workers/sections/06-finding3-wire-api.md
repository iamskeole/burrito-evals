# Finding 3: Wire API Differences

The wire API a client uses to talk to an inference backend matters more than most operators realize. On vLLM, the `/v1/responses` API introduces significantly higher error rates than `/v1/chat/completions` on multi-turn tasks.

## Chat vs Responses on vLLM

On the multi_turn_base test with fc_model=1 (schema tools):

| Wire API | Accuracy | Error Rate |
|----------|:--------:|:----------:|
| chat | 23.9% | 4.9% |
| responses | 36.1% | 28.6% |

The responses API gains accuracy but at the cost of a fivefold increase in error rate. Nearly a third of runs produce errors instead of results. The chat API is more reliable but less accurate on structured tool calls.

![Accuracy across wire APIs and backends](../burrito-evals/plots/phase_1-f01-mean_correct.png)
> Fig. 3: Mean accuracy on multi_turn_base by backend and wire API. vLLM responses API shows higher accuracy but also higher error rates.

![Error rates by backend and wire API](../burrito-evals/plots/phase_1-f05-is_error.png)
> Fig. 4: Error rates highlight the reliability gap between chat and responses APIs on vLLM.

## How Burrito Avoids This

Burrito sidesteps the chat vs responses tradeoff entirely. It accepts both APIs from clients but always communicates with the backend using `/v1/completions`. This keeps error rates low while burrito's own logic handles conversation rendering and tool call parsing.

The `/v1/completions` endpoint is simpler than chat or responses. It takes a prompt and returns tokens. Burrito builds the prompt correctly using the model's training structure, then parses the output to extract tool calls, reasoning, and final responses. This two-step approach avoids the encoding bugs that plague the higher-level APIs.

## What This Means

If you run vLLM directly and use the responses API for tool calling, expect high error rates on multi-turn tasks. The chat API is more stable but less capable with structured outputs. Neither is ideal.

The takeaway is that the simplest wire protocol often produces the most reliable results when paired with correct prompt construction on the harness side. Complex APIs add abstraction that can hide bugs in the backend's encoding logic.
