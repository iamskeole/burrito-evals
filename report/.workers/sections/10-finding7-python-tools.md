# Finding 7: Python Tool Impact

gpt-oss was trained with native python tool support. The model can write and execute code as part of its reasoning process. We test this capability on AIME25, where code execution should help with mathematical computation.

## Large Gains at Low Effort

With python enabled on AIME25:

| Effort | Accuracy Change |
|--------|:---------------:|
| Low | +21 to +24% |
| Medium | +10 to +12% |
| High | Mixed (+2.5% llama, -5.8% vllm) |

Python tools give the biggest boost at low reasoning effort, where the model does not think enough on its own to solve problems. At low effort, python execution acts as a computation engine that compensates for limited reasoning.

At medium effort, python still helps but less dramatically. The model is already doing substantial reasoning, so code execution fills gaps rather than carrying the load.

At high effort, results are mixed. The model generates enough reasoning to solve most problems without external computation. Python calls can help or hurt depending on whether the generated code is correct.

## Token Reduction

Python calls also reduce token usage at medium effort. Runs with python enabled use fewer output tokens than runs without, since the model offloads computation to code execution instead of writing out full solutions.

![Python tool impact on AIME25](../burrito-evals/plots/phase_4-f01-mean_correct.png)
> Fig. 12: AIME25 accuracy with python tool enabled vs disabled. Largest gains at low reasoning effort.

## What This Means

Python tool support is most valuable when reasoning effort is constrained. At low effort, it adds 21-24 percentage points of accuracy. At medium effort, it adds 10-12 points while reducing token usage. At high effort, the benefit disappears.

For production systems with latency or cost constraints, enabling python at medium effort gives a strong accuracy boost with lower token costs. The model uses code to verify calculations instead of generating long reasoning chains.

This finding reinforces the broader theme: the right tool configuration matters more than raw reasoning budget. Enabling python at medium effort outperforms disabling python at high effort on many problems.
