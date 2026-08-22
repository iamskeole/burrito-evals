# Executive Summary

This report evaluates gpt-oss-20b across 320,000 runs using the Big Function Calling Leaderboard, AIME25, and GPQA. We compare burrito (a model-specific inference harness) against vanilla llama.cpp and vLLM backends, testing wire APIs, function calling modes, reasoning effort levels, and native tool support.

**Configuration matters more than model capability.** The same model produces results from 0% to 83% accuracy depending on template, tool format, and API choice. Seven findings emerge from the data:

1. **The irrelevance paradox.** BFCL's irrelevance test marks any non-tool-call as correct, including silent failures. Systems that try to help score lower than systems that do nothing.
2. **The jinja fix.** Removing "commentary" from valid output channels when no tools are present moves accuracy from 3% to 40% on live tests. The default template was breaking the model.
3. **Wire API differences.** vLLM's responses API produces 28.6% error rates on multi-turn tasks. Using completions with correct prompt construction avoids the problem.
4. **Multi-turn needs schemas.** fc_model=1 (structured tool schemas) triples accuracy over fc_model=0 (AST parsing) on agentic workflows. Vanilla backends with fc_model=0 score near zero.
5. **Preserved thinking is marginal.** Slightly better with schema tools, slightly worse with AST parsing. Lower variance but not enough accuracy gain to justify added complexity for most use cases.
6. **Reasoning effort has diminishing returns.** Low to medium effort gains 35.5% accuracy on AIME25 for 4x tokens. Medium to high gains 10% for another 4x tokens. Schema tools beat more reasoning on both accuracy and cost.
7. **Python tools help most at low effort.** +21-24% accuracy at low effort, +10-12% at medium, mixed at high. Python calls also reduce token usage by offloading computation.

**Bottom line:** The model is not the bottleneck. The stack around it is. Correct templates, structured tool schemas, medium reasoning effort, and enabled python tools produce the best accuracy-to-cost ratio. Burrito addresses these configuration issues at the harness layer, with correctness gains that justify added latency for production systems.
