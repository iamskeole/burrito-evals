# Finding 2: The Jinja Fix

The default jinja chat template for gpt-oss includes "commentary" as a valid output channel even when no tools are present in the request. This single word breaks the model on nearly every non-tool-call task.

Removing "commentary" from valid channels when no tools are defined fixes the problem. The accuracy gains are dramatic.

## The Numbers

On the chat API with fc_model=0 (AST parsing, no tools in request):

| Test | Default Jinja | Fixed Jinja | Change |
|------|:------------:|:-----------:|:------:|
| live_relevance | 3.1% | 41.4% | +38.3% |
| live_simple | 3.1% | 36.7% | +33.7% |
| simple_java | 18.0% | 60.5% | +42.5% |
| simple_javascript | 13.0% | 56.5% | +43.5% |
| simple_python | 1.8% | 36.6% | +34.8% |
| multi_turn_base | 0.2% | 14.6% | +14.3% |
| irrelevance | 99.9% | 93.8% | -6.1% |

The default template produces near-zero accuracy on live and simple tests. The model hallucinates commentary channels and generates broken output. The fixed template restores functional behavior across the board.

The one metric that drops is irrelevance (99.9% to 93.8%). This connects directly to Finding 1. With the fixed template, the model actually tries to answer questions instead of producing silent failures. The irrelevance test marks those attempts as wrong, so the score goes down even though the system is doing more.

## Why This Happens

gpt-oss was trained to alternate between channels: thinking, tool calling, commentary, and final response. The default jinja template tells the model that commentary is always a valid output option. When no tools are present, the model still tries to use commentary channels, which breaks the expected response structure.

The fix is to remove "commentary" from the valid channel list when the request contains no tools. The model then produces clean responses in the expected format.

## What This Means

A one-word change in the chat template moves accuracy from 3% to 40% on live tests. This is not a minor tuning adjustment. The default template shipped with the inference stack was making the model non-functional for basic tasks.

This finding applies beyond gpt-oss. Any model trained with structured channel output is sensitive to the template it receives at inference time. Mismatches between training structure and inference template produce silent failures that look like model incompetence. If your model seems broken on simple tasks, check the template before blaming the weights.
