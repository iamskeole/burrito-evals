# Finding 4: Multi-Turn Base (Agentic Work)

Multi-turn tasks require the model to make a sequence of tool calls, process results, and continue until the task completes. These tasks expose the full gap between function calling modes and backend capabilities.

## fc_model=1 Dominates fc_model=0

The difference between schema-based tools (fc_model=1) and AST parsing (fc_model=0) is enormous on multi-turn tasks:

| Backend | fc=0 | fc=1 |
|---------|:----:|:----:|
| burrito@llamacpp | 17.2% | 52.2% |
| burrito@vllm | 17.8% | 56.2% |
| vanilla llama.cpp | ~0% | -- |
| vanilla vLLM | ~0% | -- |

With fc_model=0, vanilla backends score near zero. They cannot parse tool calls reliably enough to sustain a multi-turn conversation. Burrito with fc_model=0 achieves 17% accuracy, which is better than vanilla but still limited.

Switching to fc_model=1 triples accuracy on burrito@llamacpp (17.2% to 52.2%) and burrito@vllm (17.8% to 56.2%). Schema-based tool definitions give the model clear structure for generating and parsing tool calls across turns.

## Failures Concentrate at Turn 0

Of 42,848 total failures in multi_turn_base, 28,924 (68%) happen at turn 0. The model fails to make the first tool call correctly, and the conversation never starts. This means multi-turn accuracy is dominated by single-turn tool calling quality. Improving turn-0 reliability has outsized impact on overall multi-turn success.

![Multi-turn accuracy by backend and fc_model](../burrito-evals/plots/phase_1-f01-mean_correct.png)
> Fig. 5: Multi-turn base accuracy. fc_model=1 dramatically outperforms fc_model=0. Vanilla backends with fc_model=0 are effectively non-functional.

![Turn survival aggregation](../burrito-evals/plots/phase_5-f07-turn_survival-agg.png)
> Fig. 6: Turn survival curves. 68% of failures occur at turn 0.

## What This Means

Tool definition format is the single largest factor in multi-turn performance. Schema-based definitions (fc_model=1) give the model the structure it needs to sustain conversations. AST parsing (fc_model=0) is unreliable for multi-step workflows.

For operators building agentic systems, the lesson is clear: use structured tool schemas. The accuracy gain from fc_model=0 to fc_model=1 is larger than any backend choice.
