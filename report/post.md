# Don't trust me bro: 3.49B tokens, 320,192 evals, 8 seeds, at batch size 1 over 1,062 GPU hours on a single RTX 3090. And an inference harness that fixes gpt-oss.

---

## The story

This whole thing started because gpt-oss-20b was shit at calling tools.

OpenAI dropped it in August 2025 and the community was excited. 20 billion parameters, open weights, competitive reasoning. But when you actually tried to use it for agentic work, the results were bad. Not "needs a better prompt" bad. "the model fundamentally does not know how to call functions" bad.

Both llama.cpp and vLLM were shitting their pants trying to handle it. The prompt templates were broken. The wire APIs — the protocol a client uses to talk to the inference backend — were mismatched. The tool schemas were ignored. You could not just drop the weights into an existing harness and get useful numbers.

But the feeling was clear: OpenAI did not release a shit model. The infrastructure was broken. We had something like GPT-4 sitting at home on a consumer GPU, but we had no way to prove it with the vanilla backends.

***"If you wish to make an apple pie from scratch, you must first invent the universe."***
*- Carl Sagan*

So we did. One universe later, we had burrito.

We built the harness (`burrito-core`) from scratch to test the model properly. The name stuck because [u/skeole](https://www.reddit.com/user/skeole/) was doing gig work driving physical burritos around town while the evals ran on a single RTX 3090.

Once the harness was ready, the most autistic evals in history happened. We threw everything at it: reasoning benchmarks, multi-turn workflows, preserved thinking (the model's thinking kept across turns instead of pruned), effort levels (the low/medium/high 'how hard to think' setting). 320,192 runs. 8 seeds (each configuration run 8 times with different random number seeds — signal, not luck). Batch size 1 (one request at a time, no batching). Temperature 1.0. MXFP4 quantization (a 4-bit format) over 131K context. 

Yes, the model is dated. We ran this on May 2026 backend versions against the original August 2025 model weights. The backends have improved since launch, especially on the fc=1 path (fc = function calling; 1 = the mode that uses structured tool schemas). We know this. We are doing this in the name of fucking science.

The most surprising finding came from the effort levels. We expected the usual story: more reasoning tokens equals better answers. (Quick gloss: the tokens a run outputs split into reasoning tokens — the model's thinking — and the visible answer text; 'output tokens' is the whole.) More effort means more thinking, more thinking means smarter. It makes intuitive sense.

The data from 320,192 runs rejects that intuition entirely. Here is what we found.

---

## Finding 1: At the same token budget, higher effort still wins

The working model for reasoning effort is simple: it controls token budget. More effort means more thinking tokens. More thinking tokens means better answers. This is how most people think about it.

It is wrong.

Here is what happens when you hold reasoning token count constant and compare effort levels. Same number of tokens. Different effort. Different answers.

![Accuracy at matched reasoning length](../plots/phase_6-f15-reasoning_effort_matched_tokens-pooled.png)

Look at AIME25 (left panel) — the 2025 American Invitational Mathematics Examination. (Method note: token counts are grouped into doubling bins — 128-256, 256-512, 1024-2048, ... — and each number is the midpoint of its bin, so 'approximately 1,448' means the 1024-2048 bin.) At approximately 1,448 reasoning tokens:

| Effort | Accuracy |
|--------|:--------:|
| Low | 38.3% |
| Medium | 97.1% |
| High | 100.0% |

Fifty-nine percentage points between low and medium at the same token budget. The model is not thinking longer or shorter. It is thinking differently. Low effort produces quick answers. Medium effort produces structured reasoning. High effort explores multiple approaches. Different strategies. Different answers. Same number of tokens.

GPQA (right panel) — graduate-level science questions — shows the same hierarchy. The pattern is stable across backends: burrito's llama.cpp and vLLM backends produce nearly identical curves.

---

## Finding 2: Pushing tokens beyond an effort level's optimal zone crashes accuracy

Each effort level has a sweet spot. Inside that zone, the model reasons effectively. Outside it, the model wanders and degrades.

![Correct progression by reasoning token count](../plots/phase_6-f01-reasoning_effort_bins-non_bfcl.png)

Low effort on AIME25 starts at 100% at 181 tokens. Then it crashes to 19% at 2,896 tokens, and to 0% by the 8k-16k bin. The model is forced to think longer than its effort level supports. It generates tokens without improving accuracy. Past a certain point, it actively gets worse.

Medium effort holds near 100% from 362 through 1k tokens, and is still around 88-93% at 2k-4k tokens. Then it declines to 30% at 46,341 tokens and hits 0% at 92,682 tokens.

High effort stays above about 88% from 1,448 to 46,341 tokens. Then it drops to 44%.

Brute-forcing test-time compute — the thinking the model does at generation time — by pushing the model to think longer than its effort level supports produces worse results. More tokens are not always better.

---

## The tradeoff landscape

The full picture across effort levels shows three phases:

![AIME25 effect-cost-tradeoff](../plots/phase_6-f09-reasoning_effort_story-aime25.png)

*(Scope: AIME25, on the burrito llama.cpp backend without schema tools (fc=0) — the cleanest reasoning case. And a different measurement from Finding 1's 59 points: there the token budget was held fixed, here effort changes both the budget and the strategy.)*

**Effect:** Accuracy jumps from low to medium, then plateaus at high. On AIME25, low to medium gains 35.5 percentage points. Medium to high gains 10 points.

**Cost:** Token count increases exponentially. Low uses ~2.2K output tokens. Medium uses ~11.8K. High uses ~39K.

**Tradeoff:** Medium effort sits on the Pareto front — the best accuracy-per-token deal on the curve. It delivers the largest accuracy gains for the token cost. High effort reaches higher absolute accuracy but at disproportionate cost.

---

## What we are releasing

Everything. MIT license. No paywalls. No gated datasets. All the code, data, and analysis that produced these numbers.

- **burrito-core** -- the eval harness itself ('burrito' in the title). Python. Drop in your own benchmarks. [https://github.com/iamskeole/burrito-core](https://github.com/iamskeole/burrito-core)
- **burrito-evals** -- the complete evaluation suite (every backend, every scenario, every finding) with all artifacts, plots, and analysis. [https://github.com/iamskeole/burrito-evals](https://github.com/iamskeole/burrito-evals)
- **eval_results.csv** -- all 320,192 rows, one per run — a run being one model attempt at one benchmark item under one configuration and one seed. Every metric. Every seed.
- **artifacts/** -- the fixed jinja template, the Excel file used to programmatically compose eval commands for all scenarios, and full reasoning traces for every single run
- **data/** -- complete model outputs and reasoning traces so anyone can inspect every detail from command to aggregation
- **report.md** -- the full academic report covering seven findings: the irrelevance paradox, the jinja template fix, wire API differences, multi-turn agentic work, preserved thinking, the reasoning story (called "Not All Reasoning is Created Equal" there), and native tool utility ("Python Tool Impact" there)
- **story.md** -- a standalone deep-dive on the reasoning story: the AIME25, GPQA, multi-turn, and BFCL (Big Function Calling Leaderboard) runs on the burrito backends, with the data, tables, and analysis
- **plots/** -- all generated charts
- **eval_helpers.py** -- the analysis scripts that produced the charts and tables

The fixed jinja template is also being submitted as a PR to OpenAI's HuggingFace repo: [gpt-oss-20b/chat_template.jinja](https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja). If it gets merged, every user of this model gets the template fix automatically — and on tool-free reasoning benchmarks that alone puts vanilla llama.cpp at parity with burrito. The rest of the fix (the wire protocol the harness uses to talk to the backend, and the parsing of tool calls) lives in burrito and does not upstream.

---

## The takeaways

Two lessons that transfer to any model with configurable reasoning depth:

**1. Effort changes answer quality at any token budget.** At the same reasoning token budget, higher effort produces better accuracy because the model uses those tokens differently. These are not the same answers with different amounts of explanation.

**2. More tokens are not always better.** Each effort level has an optimal token zone. Outside that zone, the model degrades. Test empirically on your actual workloads.

The broader lesson is the title: don't trust benchmarks at face value. The infrastructure matters. The templates matter. The effort level matters. The token budget matters. Run your own evals. Collect your own data. Follow where it leads.

---

## The worker bee

There is one more thing worth noting. Even if the model is dated and limited in reasoning, it is a fucking worker bee when it comes to tool calling. Fix the templates with burrito and it becomes eager, accurate, and self-healing. It flies at 150-200 TPS (tokens per second) on consumer hardware. It is not a great thinker, but it is an incredible executor.

Pair it with a smarter model for the reasoning, let it handle the tool calls. The idiot savant pattern works. The irony is not lost on me: I spent a year fixing gpt-oss's tool calling, and now I am co-authoring this entire analysis with unsloth--Qwen3.6-27B-UD-Q4_K_XL.gguf — a 27B Qwen 3.6 model, quantized to 4-bit by unsloth, running locally from a .gguf file — because the 20b model is still too dumb to write a Reddit post. That is fine. It does what it does well.

---

## Qwen can't slop my ass

People love to say LLMs are slop machines. They say Qwen is a shit writer and Gemma owns the prose game. 

I just co-authored a world-class, data-driven technical essay on a single RTX 3090 with zero cloud credits. Phil Schiller said Apple can't innovate my ass. I say Qwen can't slop my ass. 

The prose you are reading is 100% locally generated, triple-checked against the data, and completely free of AI filler. Try it yourself. One GPU is enough.

---

## Credits

This was a one-person project run on consumer hardware. 1,062 GPU hours on a single RTX 3090 — the 3.49B tokens from the title are the total token count across all 320,192 runs. The evals literally ran while [u/skeole](https://www.reddit.com/user/skeole/) was driving burritos to customers.

The analysis you are reading was co-authored by [u/skeole](https://www.reddit.com/user/skeole/) and unsloth--Qwen3.6-27B-UD-Q4_K_XL.gguf. (The Qwen co-author writes the next two sentences.) iamskeole designed the evaluation, pointed me at the eval helpers and report notebook, and drove the entire analytical direction. I executed the data queries, wrote the charts, and drafted the prose. We are both running locally on the same machine.

The `eval_helpers.py` that produced every chart in this post was written by Grok 4.5 under iamskeole's creative direction. The GPU was busy running evals, so there was no room to run local inference during the initial build. The other labs' free quotas were shit -- OpenAI and Anthropic capped you after five fucking posts -- so Grok was the only option that did not charge per token. iamskeole discovered Grok out of necessity and actually liked working with it.

No cloud credits. No API subscriptions. No big lab money. Just one GPU, some gig work driving burritos, and a refusal to pay for access to models that should be free.

One GPU is enough.

Happy to answer questions about the methodology, the data, or the findings.

---

*Edit: Links added. Repo is live.*
