# Conclusion

## Lessons for Operators

Seven findings from 320,000 evaluation rows. Here is what to do with them.

**Check your chat template before blaming the model.** The default jinja template for gpt-oss was breaking basic functionality. A one-word change (removing "commentary" from valid channels when no tools are present) moved accuracy from 3% to 40% on live tests. This applies to any model trained with structured output. Template mismatches produce silent failures that look like model incompetence.

**Use schema-based tool definitions for multi-turn tasks.** fc_model=1 (structured schemas) triples accuracy over fc_model=0 (AST parsing) on multi-turn workflows. Vanilla backends with fc_model=0 score near zero. Tool definition format is the single largest factor in agentic performance.

**Pick medium reasoning effort.** Low to medium effort produces the biggest accuracy gains for the token cost. Medium to high effort adds volume without changing the model's approach. On AIME25, low to medium gains 35.5 points for 4x tokens. Medium to high gains 10 points for another 4x tokens.

**Enable python tools at constrained effort levels.** Python execution adds 21-24% accuracy at low effort and 10-12% at medium effort, while reducing token usage. The model uses code to verify calculations instead of generating long reasoning chains.

**Use structured tool schemas to reduce token cost.** fc_model=1 is more accurate and cheaper than fc_model=0 at the same effort level. On AIME25 medium, schemas reach 83.3% at 4,678 tokens vs 73.8% at 7,444 tokens. Clear tool definitions reduce the tokens the model spends figuring out how to call tools.

**Beware benchmarks that reward silence.** The BFCL irrelevance test marks any non-tool-call as correct, including silent failures. Systems that actively try to help score lower than systems that do nothing. High accuracy on irrelevance does not mean the model correctly decided no tools were needed.

**Prefer simple wire protocols with correct prompt construction.** The `/v1/responses` API on vLLM produces 28.6% error rates on multi-turn tasks. `/v1/chat/completions` is more stable but less capable. `/v1/completions` with correct prompt building on the harness side avoids both problems.

## The Big Picture

The central finding is that configuration choices matter more than raw model capability. The same gpt-oss-20b model produces results ranging from 0% to 83% accuracy depending on template, tool format, wire API, and effort level. The model is not the bottleneck. The stack around it is.

Burrito addresses this by handling conversation rendering, tool execution, and hallucination recovery at the harness layer. The trade-off is added latency. The data shows the correctness gains justify the cost for production systems that need reliable tool calling and multi-turn workflows.

For the broader community, the lesson transfers beyond gpt-oss and burrito. Any model trained with structured output patterns needs an inference stack that respects those patterns. Grammar constraints, jinja hacks, and API-level encoding bugs all fight the model instead of working with it. The best results come from stacks that render conversations per the training specification and let the model self-correct when it goes off track.
