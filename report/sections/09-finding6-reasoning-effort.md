# Finding 6: Not All Reasoning is Created Equal

Reasoning effort controls how much the model thinks before answering. The standard assumption is that more reasoning gives better answers at the cost of more tokens. The data shows something more nuanced: different effort levels change what the model does, not just how much it does.

## AIME25: Diminishing Returns After Medium

On AIME25 with burrito@llamacpp and fc_model=0:

| Effort | Accuracy | Output Tokens | Reasoning Tokens |
|--------|:--------:|:-------------:|:----------------:|
| Low | 38.3% | 1,954 | 1,362 |
| Medium | 73.8% | 7,444 | 6,694 |
| High | 83.8% | 30,196 | 29,574 |

Moving from low to medium effort gains 35.5 percentage points of accuracy for 4x the tokens. Moving from medium to high gains only 10 points for another 4x token increase. The jump from low to medium is where the model fundamentally changes its approach. The step from medium to high is mostly more of the same.

![AIME25 accuracy by reasoning effort](../burrito-evals/plots/phase_6-f09-reasoning_effort_story-aime25.png)
> Fig. 8: AIME25 accuracy and token usage across reasoning effort levels. Low to medium is the biggest jump.

## GPQA: Similar Pattern

On GPQA with burrito@llamacpp and fc_model=0:

| Effort | Accuracy | Output Tokens | Reasoning Tokens |
|--------|:--------:|:-------------:|:----------------:|
| Low | 55.4% | 288 | 251 |
| Medium | 67.5% | 1,968 | 1,946 |
| High | 71.9% | 17,188 | 16,938 |

Same pattern. Low to medium adds 12.1 points for 7x tokens. Medium to high adds 4.4 points for 9x tokens. The model benefits from more reasoning, but the returns drop fast.

![GPQA accuracy by reasoning effort](../burrito-evals/plots/phase_6-f10-reasoning_effort_story-gpqa.png)
> Fig. 9: GPQA accuracy by reasoning effort. Medium effort hits the sweet spot.

## Schema Tools Beat More Reasoning

The most practical finding: fc_model=1 (schema tools) consistently uses fewer tokens while achieving higher accuracy than fc_model=0 at the same effort level.

On AIME25 at medium effort, fc_model=1 reaches 83.3% accuracy using 4,678 tokens. fc_model=0 reaches 73.8% using 7,444 tokens. Schema tools are more accurate and cheaper. The model thinks less when it has clear tool definitions because it spends fewer tokens figuring out how to call tools.

![BFCL pooled reasoning effort story](../burrito-evals/plots/phase_6-f04-reasoning_effort_story-bfcl_pooled.png)
> Fig. 10: BFCL pooled accuracy by reasoning effort. Medium effort is the efficiency sweet spot.

![Matched token comparison](../burrito-evals/plots/phase_6-f15-reasoning_effort_matched_tokens-pooled.png)
> Fig. 11: Accuracy matched on token budget. fc_model=1 outperforms fc_model=0 at the same token cost.

## What This Means

Reasoning effort is not a simple knob. Low to medium effort changes the model's strategy and produces large gains. Medium to high effort adds volume without changing approach, producing small gains.

For production systems, medium effort hits the best accuracy-to-cost ratio. The extra spend on high effort buys marginal accuracy at 4-9x the token cost. And using schema-based tool definitions (fc_model=1) is a better investment than cranking up reasoning effort, since it improves accuracy while reducing token usage.
