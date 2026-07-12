# burrito-evals

This repo contains the raw eval results for gpt-oss-20b, an open-weights model released by OpenAI in August 2025. The model was evaluated on BFCL V4, as well as some of the benchmarks OpenAI themselves published when releasing the model: AIME25 and GPQA.

### tree -L2
```bash
.
├── data
│   ├── bfcl_v4
│   ├── eval_results_all.csv
│   ├── eval_results_bfcl_v4.csv
│   ├── eval_results_gpt_oss.csv
│   ├── gpt_oss
│   └── README.md
├── eval_aggregator.py
├── eval_helpers.py
├── eval_report.ipynb
├── eval_report.md
├── eval_specs.xlsx
├── eval_versions.txt
├── LICENSE
├── __pycache__
│   └── eval_helpers.cpython-312.pyc
├── pyproject.toml
├── README.md
├── scripts
│   ├── install_nvidia_power_service.sh
│   ├── nvidia-power.service
│   ├── serve_llamacpp_default_jinja.sh
│   ├── serve_llamacpp_fixed_jinja.sh
│   ├── serve_vllm_default_jinja.sh
│   └── serve_vllm_fixed_jinja.sh
└── uv.lock

6 directories, 21 files
```
### df.head()

```python
                                            run_name           backend  \
0  burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867  burrito@llamacpp   
1  burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867  burrito@llamacpp   
2  burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867  burrito@llamacpp   
3  burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867  burrito@llamacpp   
4  burrito@llamacpp_chat_medium_t-1_b-1_f-0_s-680867  burrito@llamacpp   

  wire_api        model_name  fc_model  browser_enabled  python_enabled  \
0     chat  gpt-oss-20b-chat         0                0               0   
1     chat  gpt-oss-20b-chat         0                0               0   
2     chat  gpt-oss-20b-chat         0                0               0   
3     chat  gpt-oss-20b-chat         0                0               0   
4     chat  gpt-oss-20b-chat         0                0               0   

   batch_size    seed reasoning_effort  ...            test_id  \
0           1  680867           medium  ...  multi_turn_base_0   
1           1  680867           medium  ...  multi_turn_base_1   
2           1  680867           medium  ...  multi_turn_base_2   
3           1  680867           medium  ...  multi_turn_base_3   
4           1  680867           medium  ...  multi_turn_base_4   

         test_name   test_type input_token_count  output_token_count  \
0  multi_turn_base  multi_turn           83338.0              4252.0   
1  multi_turn_base  multi_turn           37632.0              1543.0   
2  multi_turn_base  multi_turn          189586.0              5411.0   
3  multi_turn_base  multi_turn            8897.0               950.0   
4  multi_turn_base  multi_turn           23660.0               563.0   

     latency  is_error  correct  n_tool_calls_browser  n_tool_calls_python  
0  45.435766         0      0.0                     0                    0  
1  17.716507         0      1.0                     0                    0  
2  75.694669         0      0.0                     0                    0  
3   7.828380         0      0.0                     0                    0  
4   8.746322         0      0.0                     0                    0  

[5 rows x 21 columns]
```

### df["backend"].unique()
```python
<StringArray>
[
    'burrito@llamacpp',
    'vllm',
    'burrito@vllm',
    'llamacpp@default-jinja',
    'llamacpp@fixed-jinja'
]
Length: 5, dtype: str
```

# Different backends evaluated

- llamacpp@default-jinja (vanilla llama.cpp execution)
- llamacpp@fixed-jinja (vanilla llama.cpp execution but with a custom jinja template that fixes prompt based tool calling, more on that later)
- vllm (vanilla vLLM execution)
- burrito@llama.cpp (custom inference harness using llama.cpp next-token prediction on /completions endpoint)
- burrito@llama.cpp (uses vLLM for next-token prediction on /completions)

# The burrito inference harness

A custom harness built from the ground up to:

- fix tool calling as both llama.cpp and vLLM had reports and erroneous tool calls; it does this by treating the model as an intelligent agent as opposed to a mathematical artifact, eg in the case of tool name hallucinations, it catches that and prompts the model to correct itself, or in the case of doom-loops it also detects that and again prompts the model to try a different approach; this is in stark contrast to eg llama.cpp that uses grammar tricks to bias logits when there are tools present in the incoming request at the cost of disabling builtin python and browser tools as these function as their individual namespaces and llama.cpp hardcodes a 'functions' namespace the model is allowed to use

- enable native tools gpt-oss was trained to use (stateful, sandboxed jupyter kernel python execution and a browser tool to search and browse the web)

- be a drop in replacement *in front* of vanilla backends, eg call http://burrito/v1/chat/completions instead of http://llamacpp/v1/chat/completions; burrito renders prompts, user, assistant and tool messages correctly using the Harmony response format the model was trained to use, then talks to either llama.cpp or vLLM in the backend for raw next-token predictions on the properly formatted messages using /completion endpoints (think dropping to assembly but for tokens)

- enable per-request reasoning effort configuration

# Model setup

BFCL tests run two variations: FC and not FC; FC means the backend is presented with a list of tools in OpenAI compatible format and needs to inject into system prompt; non FC means the backend isn't presented with tool schemas separately, but the test harness builds the prompt itself and then expects model to call tools in a specific format and then uses AST to parse model outputs.

# Two wire APIs

OpenAI compatible chat completions (/v1/chat/completions) and Responses API (/v1/responses) endpoints, as there have also been reports of inconsistencies across the two wire api formats, and OpenAI has already transitioned away from Chat Completions towards Responses API as the de facto wire api for most products.

# Eight seeds

OpenAI recommends running the model at temperature 1.0, so the evals use this default. In order to both control for determinism / reproducibility as well as test for cross-seed consistency given real world deployments would not fix seeds but are recommended to use t=1.0, each test is ran 8 times, on 8 different seeds.

# Reasoning effort

The model was trained to support three reasoning efforts: low, medium and high. However, llama.cpp only provides that as a jinja variable customization at startup so all requests in that server instance use the reasoning effort configured on boot. vLLM offers per-request customization but suffers from tool calling and wire api inconsistencies. To enable a baseline, most tests were conducted on the 'medium' reasoning effort, with one specific exception: burrito@llamacpp and burrito@vllm test for low and high; assuming a baseline of correctness was established on medium reasoning using burrito, the intent is to investigate deltas in performance coming from different reasoning efforts. All other benchmark combinations are executed on default medium reasoning allowing for accurate and fair comparison between backends / harnesses.

# Original Research questions

1. Does burrito provide a statistically significant lift in model accuracy? 

2. Does quantization / tensor format impact model accuracy? Eg. *.gguf files for llamacpp vs. 'factory defaults' in *.safetensors for provider-defaults in MXFP4 format.

3. Does reasoning effort materially impact model performance?

4. Do native tools (python, browser) help model improve baseline performance? These tests were only ran on the gpt_oss test suite (AIME25 and GPQA) and only for the burrito options as the other backends do not allow for this functionality.

5. Does burrito materially impact end to end response times? NB: latency, as measured in the benchmarks is end to end, in seconds, from prompt submit to answer complete.

5. Which is the optimal backend / harness to run in production? (eg how does the pareto frontier / magic quadrant look like for the different backends / harnesses?)

> All questions need to be answered both in aggregate across all tests executed, as well as in detail at the test type level (eg bfcl_v4 vs gpt_oss / aime / gpqa) and test_name level (eg individual tests within a category).