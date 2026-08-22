good job, i think we're good with report.md as the "academic" version.

now we'll need to create a file story.md next to it, where we focus on this story exclusively as i think that's the real story from the entire dataset. we'll need to do this in excruciating detail, with charts, tables, the works. triple checking everything, making sure the story really shines and screams. world class data analysis, world class copywriting and galactic storytelling.

HOWEVER! we're reaching the end of the context limit in this session, so we'll be hit with compaction soon. therefore, as a next step, i need you to leave detailed notes for your future self on where we are and where we're going, at the top of PLAN.md, so that we can use that to kick off after compaction.

ALSO! if helpful, this is the original user request that started this entire exercise:

<--- BEGIN ORIGINAL USER REQUEST --->

Ok so here's the deal. Almost one year ago, to the date, OpenAI released their first open-weights model since GPT-2: gpt-oss. The smaller of the two, gpt-oss-20b was meant for consumer GPU deployment, whereas the larger, gpt-oss-120b was aimed more at pro workstations given the 120 billion parameter size.

Community reception was.. bleak. Everybody was pretty much shitting on the model for two reasons: it was (somewhat) censored, refusing to engage in a lot of work and always spending a ton of tokens questioning itself based on some user-invisible policy it was trained on and whether the policy allows it to answer. While a valid criticism, the model did less to none of this for most unquestionable tasks. The second reason was very poor performance around tool calling, even though OpenAI was specifically making a point about how great the model was for this in their announcements.

Playing around with the model at that time, I quickly managed to confirm both issues flagged by the community. But something else as well: at around 15GB used for both the model and full size context (131072), it was the first time consumers (aka GPU poor) could use such capacity on local, relatively modest hardware. Additionally, when ran on a higher end GPU such as the RTX 3090, the model was flying at 200+ tokens per second, which was totally unheard of for unquantized / unlobotomized models.

Testing it in all sorts of regular conversations, the model also seemed fairly smart for its size, almost like GPT-4 (ish) levels of intelligence on my own hardware.

Coupled with the fact that the model had been trained to use two "native" tools, a python interpreter and a browser, this all in theory made for a *very* interesting proposition for consumers.

Except the community feedback: model is shit.

So I thought to myself this can't be right, OpenAI doesn't do shit models. Digging deeper into the docs and repos shared by OpenAI, I learned about the Harmony response format they had used to train the model.

From their docs:

```python
<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>
```

That's an example of how messages are structured, introducing the concept of channels for various message rendering dynamics. That was a first lightbulb moment. I saw this as a *very* elegant way to handle messaging metaphors in what's still essentially a stream of tokens / integers. But more than this, I figured a lot of thought and engineering went into coming up with this concept, so they couldn't have just invented it for a model they may or may not update after the initial open-weights release. It's very likely this was already something they were using in the bigger brother ChatGPT & co, advancing beyond the limits of what the consumer / LocalLlama community was using (jinja chat templates).

Case in point, close to a year later, Google launches the Gemma 4 series of model and here's an extract from *their* docs:

```python
<|turn>system
<|think|>You are a helpful assistant.<|tool>declaration:get_current_temperature{...}<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|channel>thought
...
<channel|><|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

Look familiar? 

Well, I couldn't foresee the Gemma 4 future at the time gpt-oss was released, but I think this only reinforces I was on to something with my thinking back then.

Anyway, back to gpt-oss, this seemed to be a case of "you're holding it wrong". So I tried both "vanilla" backends (llama.cpp, vLLM) for both wire apis they were exposing at the time: /v1/chat/completions and the newer /v1/responses API that OpenAI has since almost entirely switched to. Additionally, when reading more into the docs for the Responses API and how it builds objects and streams, it seemed to map relatively nicely back to the channel concepts found in Harmony (Exhibit B I guess?).

The horror followed. Both llama.cpp and vLLM were utterly broken when it came to tool calling but also general chat requests. Responses API was barely functional in vLLM and non-existent in llama.cpp (at that time, now both support both wire APIs.. somewhat). Additionally, there was no way to test for model behavior when using native tools it was trained with since llama.cpp was (and still is) hardcoding a 'functions.' namespace for all tools it receives in the request body and both python and browser tools are trained into the model to act as their own namespaces, so it's virtually impossible to use as the model was trained to use them. And vLLM was shipping some demo servers for both python and browser, but reliant on third party API keys for browser search and retrieval, which was in my book against the ethos of sovereign solutions. Oh, and the vLLM implementation of the Harmony response format was broken such that probably 9/10 calls (even regular chat calls, no tools) to that model were breaking.

So there I am trying to answer two simple questions:

1. is there a significant difference in model performance between llama.cpp (GGUF weights) vs vLLM (safetensors)? This was the first model in this class that consumers could run on a 24GB card AND use batching in what was the "official" off the assembly line format - safetensors natively quantized to MXFP4. Was that *the* way to run it vs. single threaded llama.cpp? 

2. is the model DOA for any sort of agentic task? Tool calls were failing miserably on both backends while the creators were making a big deal around the ability and training to natively call tools.

3. how much does the model improve in its capacity as both a personal assistant but most importantly agentic coder when given access to the two native tools it had been trained with (python and browser).

Given both backends were shitting their pants when handling that model, the answer was the case of the infamous meme: "if you wish to make an apple pie from scratch, you must first invent the universe".

So here we are, one universe later.

The dir we're in hosts two repos: `burrito-core` and `burrito-evals`.

`burrito-core` is my answer to the universe creation problem, and the first step I had to take in order to answer the three damn questions that were bugging me almost a year ago. It's technically a proxy that sits between the caller and either llama.cpp or vLLM. It captures the user prompt, renders all messages in the conversation 100% to the Harmony spec, and then uses either "vanilla" backend (interchangeably) on the old and tested /completions endpoint for raw, token by token predictions. It intercepts model hallucinations, doom loops and all sorts of failure modes (which honestly may be a by-product of the 20B size, I don't know how the 120B behaves since I don't have the hardware to test it), and "talks" back to the model on failures to help it course correct before returning final answers to the user. E.g., where the "vanilla" backends (or mostly llama.cpp) treat the model as a mathematical artifact and enforce certain structure by manipulating logits with all sorts of sampling params or grammars, burrito treats the model as an intelligent artifact and pretty much tells it "bro, you're doing it wrong, get your shit together". And on top of this it also enables both a python Jupyter sandbox and a browser engine specifically tailored around local / self hosting via SearXNG. In building it I've put a lot of focus on agentic type work, hence one of the key evals I think is the multi_turn_base test category in the BFCL benchmark suite (and specifically split between fc_model 0 and 1 which is a prompt based AST parsing vs. schema tool definition). 

Also, I think there's a really nice paradox I've stumbled upon: the "irrelevance" (test category) paradox. I think BFCL irrelevance scores any type of answer as correct as long as it's NOT calling one of the tools, even server errors! This means that in some cases, e.g., if the server returns an error, the AST parser looks at the error string, sees it's not a tool call and says ok boss, good to go you provided a correct answer, which is.. wrong, hence the paradox.

Also, throughout this entire journey I implemented a borderline stupid / dead simple fix to the model's jinja template (you'll see some runs called llamacpp@fixed-jinja which use this); in short, it's a dead simple fix / discovery I made in the process of building burrito. llamacpp / the default OpenAI jinja always defaults to something like 'valid channels: analysis, commentary, final' in the system prompt. But when there are no tools in the request body, eg it's AST / prompt base tool calling, not structured functions, and the caller parses model output itself, if you leave that same message in the system prompt the model tends to hallucinate a commentary channel (where RL trained it to issue tool calls) and everything breaks. So I updated the jinja template to do a check for tools and if no tools 'valid channels: analysis, final'. That's it. Dead simple and borderline stupid and I'll submit a PR with that fix to the official HuggingFace repo / chat template of the model too.

`burrito-evals` is the result of over 2 billion tokens, 300k+ questions for close to 900 hours clock time running on my own RTX 3090 test both the vanilla backends and the harness I built.

Which brings us to my current conundrum. Evals are ready and I think my last step is to create a report to tell the story of the data.

But here's the deal -- we're one year past go. Life found a way to get in the way around September last year, so I did all this while driving physical burritos around town, in parking lots and at night. The name kinda stuck in that irony. Thing is, the model itself is likely no longer relevant and (so far) has not been refreshed by OpenAI. I still think it may have use cases when paired with other newer, more intelligent models such as the recent Gemma 4 31B or Qwen 3.6 27B, but used as an executor since it's very eager to do work and call tools, but it's I guess similar to the idiot savant concept, has good inherent intelligence but dumb as a rock in terms of creativity, nuance etc.

So in short, while I think there may be some merit to the model itself, it may be just in my mind or frankly none at all given the latest model releases and the model itself may as well be ignored at this point. So while burrito as a harness may also be less relevant at this point, what I think has merit is everything I've discovered, researched and evaluated during this time. So I want to share both the harness and the evals with the community, freely, openly, under an MIT license.

But here's the crux. Since we're beyond one year later after the original model release, I think the actual harness itself is probably not that useful unless for inspiration for others, since the model may not be top of mind nor is it topping any benchmarks with those that followed since its original release.

But I think there's a story here that's worth sharing with the world, that can be applicable / transferable to other models, specifically:

1. the irrelevance paradox - benchmarks can silently fail or promote model performance
2. the stupid jinja fix - sometimes it pays off to dig deeper into infra
3. preserved thinking is not a silver bullet - openai recommends pruning thinking history unless the model is still issuing tool calls, we tested non-pruning (burrito-pt* versions) and accuracy only marginably increases, not sure even if statistically significant; variance does seem to decrease though
4. not all reasoning is created equal - i think this is the biggest story that can transfer to other models, eg for the same token budget across low, medium, high efforts, the actual answers seem entirely different and translate across backends and resulting accuracies

And this is where you come in. Take a poke, explore a little in each dir in the repo if this backstory is not enough.

Then have a look in burrito-evals/data/eval_results.csv. That's the file that aggregates the full dataset of all evaluations I ran.

Then have a look at burrito-evals/eval_helpers.py. These are some helpers we've put together to help generate plots.

Finally, I've already generated many plots in burrito-evals/plots that i think are relevant using burrito-evals/eval_report.ipynb, using the helpers noted earlier. You can either inspect the plot images themselves but probably also a good idea to run the actual helper methods you see in the eval_report.ipynb cells, since every method also returns the data it uses in the plot and it may be better to work with raw data than the resulting image.

So I need your help with a report type story. Data driven that tells the story of the most fascinating lessons learned from building and evaluating burrito, that can serve as future reference points for the community to build their understanding around, at least when it comes to other models. 

Your task is to produce a single, self-contained, Markdown file report that's this entire story, driven by the data (tables, charts, academic style but again focused on digging up the diamonds in the rough). You can either link plots from burrito-evals/plots or generate new ones if you find a different angle.

I don't know, I think that's the best I can explain it, I guess I think there's some value hidden in there somewhere even if the model is no longer relevant and need your help to dig it out so we can share it with the world.

Couple of pointers: 

(1) create a workspace in the root of the repo that you can then 
(2) use to manage your context window intelligently and finally
(3) create a PLAN.md file in the workspace where you decompose your work into pieces; this should be your first deliverable before yielding the turn back to me; then we'll start a new session and I'll prompt you to go through each of the plan items until we reach the final deliverables
(4) be mindful of recursion, e.g., ls -R will turn up huge lists of files since the `burrito-evals` also contains the full inference traces for all 300k+ questions, so be very careful of how you load that data into your context window
(5) as you progress, return to the PLAN.md file every now and then to make sure it is as close as possible to the current state of your work, so we can use it efficiently to recover from context compaction

Also, there's probably a lot to digest and you have a limited context window, so it might be a good idea to build scripts that you can (re)use, as well as leave yourself some breadcrumbs / status / successes and failures etc. for the recoveries that may be needed upon context compaction. In other words, don't try to "do it all in your head", work smart, not slick. This is probably a large exercise and we'll likely need to come back to different parts of it as we work, so it's probably a good idea to work in a dedicated dir / workspace and save stuff as we go along.

Over to you, chief, and thanks!

---
misc notes

* ensure enough talk / granularity about wire api differences (vLLM is really, really bad on responses api) and a clearer picture on multi turn base which is the closest thing to agentic work irl, split properly between fc_model 0 and 1
* if it helps, i ran this in 4 phases: 

1. **Can the model function-call reliably across backends?** — BFCL v4 across all 5 backends, 2 wire APIs, 2 FC modes, fixed `reasoning_effort=medium`.

2. **What's the vanilla reasoning/QA baseline?** — All backends on AIME25 and GPQA Diamond, `fc_model=0`, `reasoning_effort=medium`, `responses` API.

3. **Does reasoning effort change anything?** — `burrito` only, `low`/`high` across BFCL v4 and the reasoning/QA suite. (`medium` already covered in Phases 1–2.)

4. **Do model-native tools help?** — `burrito` only, `python_enabled=1` on AIME25. (`browser` tests were killed; see below.)

phases 5 and 6 are deep dives into the reasoning story

(so only phase 1 was matched across all variations, to establish a baseline; when that was established, i no longer ran chat wire api and some others since i was running this on batch size 1 on a single rtx 3090 and i wanted to use compute efficiently)
* the reason for multiple seeds (in addition to establishing variance stability) was that i used temperature 1.0 on purpose, since that's the producer recommended setting and i wanted to have as realistic / close to real deployment setup as possible, hence no other / lower temperatures

* on the browser_enabled 0; i actually killed the browser tests; as i was testing i ran into another erm.. paradox? knowledge websites (the ones the model kept wanting to open, even wikipedia at times) seem to be falling prey to knowledge capitalism, locked behind paywalls or antibot measures only open to the big players; also, search results with searxng and i guess in general weren't deterministic (and i went thru great lengths to make all this data deterministic and reproducible, vanilla servers are running on gimped settings that disable prompt caching etc, hence the ~100-ish tokens per second vs normally 200+); so i decided there's no value in testing the browser tool; although it works, and in general non-capitalistic setups (eg general assistant, looking up news etc) it works well (except reuters that always blocks); so the tool is there, the world just sucks

* ensure any text in final deliverables is free from ai slop, eg the inverse / surprise reveals, the not x but ys, the em dashes etc, i'd appreciate something more human(e)

<--- END ORIGINAL USER REQUEST --->