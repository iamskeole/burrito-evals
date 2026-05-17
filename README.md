baseline
- medium reasoning so i can automate a good part of it, otherwise have to keep relaunching llama.cpp server (and vllm not tractable, explain why)
- t=0, s=69421337
- special server startup commands for both llama.cpp and vllm to ensure deterministic outputs (as much as possible..)
- only irrelevance and simple_python for bfcl, otherwise multiseed baloons runtime

prodlike
- 8 seeds, t=1, top_k=64, min_p=0.0, top_p=1.0
- medium reasoning for baseline (no tools, burrito vs direct)
- then scaled for burrito@{backend} to show (a) different reasoning and (b) impact of native tools
