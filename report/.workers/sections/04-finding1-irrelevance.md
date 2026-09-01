# Finding 1: The Irrelevance Paradox

The BFCL irrelevance test measures whether a model correctly refrains from calling tools when none are relevant. The scoring rule is simple: any response that does not contain a tool call counts as correct.

This rule creates a blind spot. The test rewards inaction. A model that correctly decides no tools are needed scores the same as a model that crashes, produces empty output, or never tries. The test cannot tell the difference.

## The Data

Across all backends, error rates on the irrelevance test are near zero. Accuracy scores range from 87% to 93%. The failure mode breakdown reveals the structure:

- 47,568 responses marked `success` (correct = 1)
- 267 infrastructure errors also scored correct (correct = 1) because they produced no tool call
- 5,925 model failures scored incorrect (correct = 0) because the model tried to call a tool when it should not have

The 267 infra errors scored as correct are the silent failures the test cannot detect. They are a small fraction (0.5% of total), but they illustrate the structural problem: the benchmark has no way to verify that a non-tool-call response came from correct judgment rather than a broken system.

The larger signal is in the incorrect responses. Burrito backends produce far more of them: 1,959 for burrito@llamacpp and 2,051 for burrito@vllm, compared to 558 for llamacpp@default-jinja and 563 for vLLM. Burrito tries harder. When the model attempts a response and produces a tool call on a question that requires none, the benchmark marks it wrong. Vanilla backends with the default jinja template often produce no output at all, which the benchmark reads as correct.

This is why burrito scores lower on irrelevance (87%) than vanilla backends (93%). Lower irrelevance accuracy means the system is more active, not less capable.

![Mean accuracy on the irrelevance test by backend](../burrito-evals/plots/phase_1-f01-mean_correct.png)
> Fig. 1: Mean accuracy on irrelevance test by backend. Burrito backends score lower because they attempt tool calls more often, and incorrect tool calls are penalized.

![Error rates on the irrelevance test](../burrito-evals/plots/phase_1-f05-is_error.png)
> Fig. 2: Error rates on irrelevance test are near zero across all backends.

## What This Means

The irrelevance test conflates correct restraint with system failure. A high score can come from the model correctly deciding no tools are needed, or from the system producing no output at all. The benchmark cannot distinguish between the two.

This is a warning for anyone using BFCL or similar benchmarks. A test that marks non-tool-calls as correct rewards silence. Systems that actively engage with the task will score lower, even when their behavior is more useful in practice.
