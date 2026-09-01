# Finding 5: Preserved Thinking

Preserved thinking (pt) is a burrito variant that keeps the model's internal reasoning visible in the output rather than stripping it. The hypothesis is that preserving reasoning chains helps the model maintain coherence across turns and reduces variance.

## Mixed Results on Multi-Turn

On multi_turn_base at medium reasoning effort:

| Backend | fc=0 | fc=1 |
|---------|:----:|:----:|
| burrito-pt@llamacpp | 15.2% | 55.1% |
| burrito@llamacpp | 17.2% | 51.8% |

With fc_model=0, preserved thinking is slightly worse (15.2% vs 17.2%). With fc_model=1, preserved thinking is slightly better (55.1% vs 51.8%). The direction flips depending on tool calling mode.

Standard deviation is lower for preserved thinking versions, indicating more consistent performance across seeds. The model is less likely to have outlier good or bad runs.

![Preserved thinking accuracy on multi-turn](../burrito-evals/plots/phase_5-f01-mean_correct.png)
> Fig. 7: Preserved thinking vs standard burrito on multi-turn base. Gains are marginal and direction depends on fc_model.

## Minimal Impact on BFCL Tests

On single-turn BFCL tests (non-multi-turn, medium effort), preserved thinking shows small deltas that flip in direction across different tests. The effect is not consistent enough to recommend pt as a general improvement.

## What This Means

Preserved thinking is not a silver bullet. It produces marginally better results with schema tools (fc=1) and marginally worse results with AST parsing (fc=0). The lower variance is a real benefit for production systems that need predictable performance.

The trade-off is complexity. Preserved thinking adds processing overhead and changes the output format. For most use cases, the accuracy gains do not justify the added complexity. Systems that need consistent performance across runs may find the lower variance worth it.
