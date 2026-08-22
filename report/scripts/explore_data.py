#!/usr/bin/env python3
"""
explore_data.py
Extract key statistics and findings from eval_results.csv for the report.
Run this to get a summary of all key data points needed for each report section.
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/p/code/local/burrito-evals')

CSV = '/home/p/code/local/burrito-evals/data/eval_results.csv'

def load():
    return pd.read_csv(CSV, low_memory=False)

def section_1_irrelevance(df):
    """The Irrelevance Paradox -- BFCL irrelevance test marks non-tool-calls as correct."""
    print("=" * 70)
    print("SECTION 1: THE IRRELEVANCE PARADOX")
    print("=" * 70)

    irr = df[df['test_name'] == 'irrelevance']
    print(f"\nTotal irrelevance rows: {len(irr)}")
    print(f"Unique backends: {sorted(irr['backend'].unique())}")
    print(f"Unique wire APIs: {sorted(irr['wire_api'].unique())}")
    print(f"Unique fc_models: {sorted(irr['fc_model'].unique())}")

    # Accuracy by backend (mean across all runs)
    print("\n--- Mean accuracy by backend (irrelevance test) ---")
    acc = irr.groupby('backend')['correct'].mean() * 100
    for b, v in acc.sort_values(ascending=False).items():
        print(f"  {b}: {v:.1f}%")

    # Error rate by backend
    print("\n--- Error rate by backend (irrelevance test) ---")
    err = irr.groupby('backend')['is_error'].mean() * 100
    for b, v in err.sort_values(ascending=False).items():
        print(f"  {b}: {v:.1f}%")

    # The paradox: high accuracy + high error rate
    print("\n--- The paradox: accuracy + error rate side by side ---")
    combined = irr.groupby('backend').agg(
        accuracy=('correct', 'mean'),
        error_rate=('is_error', 'mean'),
        n=('correct', 'count')
    )
    combined['accuracy_pct'] = (combined['accuracy'] * 100).round(1)
    combined['error_pct'] = (combined['error_rate'] * 100).round(1)
    print(combined.to_string())

    # Error types in irrelevance
    print("\n--- Error types in irrelevance (non-zero) ---")
    irr_errors = irr[irr['is_error'] == 1]
    if len(irr_errors) > 0:
        print(irr_errors['error_type'].value_counts().head(20))
    else:
        print("No errors in irrelevance test")

    # Failure modes
    print("\n--- Failure modes in irrelevance ---")
    print(irr['failure_mode'].value_counts())

def section_2_jinja_fix(df):
    """The Jinja Fix -- default vs fixed jinja template comparison."""
    print("\n" + "=" * 70)
    print("SECTION 2: THE JINJA FIX")
    print("=" * 70)

    # Compare default-jinja vs fixed-jinja on chat API, fc_model=0
    dfl = df[df['backend'] == 'llamacpp@default-jinja']
    dff = df[df['backend'] == 'llamacpp@fixed-jinja']

    # Focus on chat API, fc_model=0 (where the fix matters most)
    dfl_chat_f0 = dfl[(dfl['wire_api'] == 'chat') & (dfl['fc_model'] == 0)]
    dff_chat_f0 = dff[(dff['wire_api'] == 'chat') & (dff['fc_model'] == 0)]

    print("\n--- Jinja fix impact: default vs fixed, chat API, fc_model=0 ---")
    for test in sorted(dfl_chat_f0['test_name'].unique()):
        d_acc = dfl_chat_f0[dfl_chat_f0['test_name'] == test]['correct'].mean() * 100
        f_acc = dff_chat_f0[dff_chat_f0['test_name'] == test]['correct'].mean() * 100
        d_err = dfl_chat_f0[dfl_chat_f0['test_name'] == test]['is_error'].mean() * 100
        f_err = dff_chat_f0[dff_chat_f0['test_name'] == test]['is_error'].mean() * 100
        print(f"  {test:20s}: default={d_acc:5.1f}% (err {d_err:5.1f}%)  fixed={f_acc:5.1f}% (err {f_err:5.1f}%)  delta={f_acc-d_acc:+.1f}%")

    # Also compare on responses API
    print("\n--- Jinja fix impact: default vs fixed, responses API, fc_model=0 ---")
    dfl_resp_f0 = dfl[(dfl['wire_api'] == 'responses') & (dfl['fc_model'] == 0)]
    dff_resp_f0 = dff[(dff['wire_api'] == 'responses') & (dff['fc_model'] == 0)]
    for test in sorted(dfl_resp_f0['test_name'].unique()):
        d_acc = dfl_resp_f0[dfl_resp_f0['test_name'] == test]['correct'].mean() * 100
        f_acc = dff_resp_f0[dff_resp_f0['test_name'] == test]['correct'].mean() * 100
        d_err = dfl_resp_f0[dfl_resp_f0['test_name'] == test]['is_error'].mean() * 100
        f_err = dff_resp_f0[dff_resp_f0['test_name'] == test]['is_error'].mean() * 100
        print(f"  {test:20s}: default={d_acc:5.1f}% (err {d_err:5.1f}%)  fixed={f_acc:5.1f}% (err {f_err:5.1f}%)  delta={f_acc-d_acc:+.1f}%")

def section_3_wire_api(df):
    """Wire API Differences -- chat vs responses API across backends."""
    print("\n" + "=" * 70)
    print("SECTION 3: WIRE API DIFFERENCES")
    print("=" * 70)

    # Phase 1 data: medium effort, non GPT-OSS tests
    p1 = df[(df['reasoning_effort'] == 'medium') & (df['test_type'] != 'GPT-OSS')]

    for backend in sorted(p1['backend'].unique()):
        bk = p1[p1['backend'] == backend]
        print(f"\n--- {backend} ---")
        for api in sorted(bk['wire_api'].unique()):
            for fc in sorted(bk[bk['wire_api'] == api]['fc_model'].unique()):
                subset = bk[(bk['wire_api'] == api) & (bk['fc_model'] == fc)]
                acc = subset['correct'].mean() * 100
                err = subset['is_error'].mean() * 100
                print(f"  {api:10s} fc={fc}: accuracy={acc:5.1f}%  errors={err:5.1f}%  n={len(subset)}")

    # vLLM specific: chat vs responses comparison
    print("\n--- vLLM deep dive: chat vs responses ---")
    vllm = df[df['backend'] == 'vllm']
    for api in sorted(vllm['wire_api'].unique()):
        for fc in sorted(vllm[vllm['wire_api'] == api]['fc_model'].unique()):
            subset = vllm[(vllm['wire_api'] == api) & (vllm['fc_model'] == fc)]
            for test in sorted(subset['test_name'].unique()):
                t = subset[subset['test_name'] == test]
                acc = t['correct'].mean() * 100
                err = t['is_error'].mean() * 100
                print(f"  {api:10s} fc={fc} {test:20s}: acc={acc:5.1f}%  err={err:5.1f}%")

def section_4_multi_turn(df):
    """Multi-Turn Base -- agentic work, split by fc_model."""
    print("\n" + "=" * 70)
    print("SECTION 4: MULTI-TURN BASE (AGENTIC WORK)")
    print("=" * 70)

    mt = df[df['test_name'] == 'multi_turn_base']
    print(f"\nTotal multi_turn_base rows: {len(mt)}")

    # Accuracy by backend and fc_model
    print("\n--- Mean accuracy by backend and fc_model ---")
    for backend in sorted(mt['backend'].unique()):
        bk = mt[mt['backend'] == backend]
        for fc in sorted(bk['fc_model'].unique()):
            for api in sorted(bk[bk['fc_model'] == fc]['wire_api'].unique()):
                subset = bk[(bk['fc_model'] == fc) & (bk['wire_api'] == api)]
                acc = subset['correct'].mean() * 100
                err = subset['is_error'].mean() * 100
                avg_turns = subset['mt_num_turns_success'].mean()
                print(f"  {backend:30s} fc={fc} {api:10s}: acc={acc:5.1f}%  err={err:5.1f}%  avg_success_turns={avg_turns:.1f}")

    # Turn survival stats
    print("\n--- Turn survival stats (success_pct by turn) ---")
    # For multi-turn, look at mt_num_turns_success_pct
    mt_success = mt[mt['correct'] == 1]
    print(f"Successful runs: {len(mt_success)}")
    if len(mt_success) > 0:
        print(f"  Mean success turns: {mt_success['mt_num_turns_success'].mean():.1f}")
        print(f"  Mean total turns: {mt_success['mt_num_turns_total'].mean():.1f}")
        print(f"  Mean success pct: {mt_success['mt_num_turns_success_pct'].mean():.1f}%")

    # Failed turn analysis
    mt_failed = mt[mt['correct'] == 0]
    print(f"\nFailed runs: {len(mt_failed)}")
    if len(mt_failed) > 0:
        print(f"  Mean failed turn idx: {mt_failed['mt_failed_turn_idx'].mean():.1f}")
        print(f"  Failed turn idx distribution:")
        print(mt_failed['mt_failed_turn_idx'].value_counts().sort_index())

def section_5_preserved_thinking(df):
    """Preserved Thinking -- burrito-pt vs burrito comparison."""
    print("\n" + "=" * 70)
    print("SECTION 5: PRESERVED THINKING")
    print("=" * 70)

    # burrito-pt backends vs burrito backends on multi_turn_base
    pt_llama = df[(df['backend'] == 'burrito-pt@llamacpp') & (df['test_name'] == 'multi_turn_base')]
    pt_vllm = df[(df['backend'] == 'burrito-pt@vllm') & (df['test_name'] == 'multi_turn_base')]
    b_llama = df[(df['backend'] == 'burrito@llamacpp') & (df['test_name'] == 'multi_turn_base') & (df['reasoning_effort'] == 'medium')]
    b_vllm = df[(df['backend'] == 'burrito@vllm') & (df['test_name'] == 'multi_turn_base') & (df['reasoning_effort'] == 'medium')]

    print("\n--- Preserved thinking impact on multi_turn_base ---")
    for name, data in [("burrito-pt@llamacpp (medium)", pt_llama[pt_llama['reasoning_effort'] == 'medium']),
                       ("burrito-pt@vllm (medium)", pt_vllm[pt_vllm['reasoning_effort'] == 'medium']),
                       ("burrito@llamacpp (medium)", b_llama),
                       ("burrito@vllm (medium)", b_vllm)]:
        for fc in sorted(data['fc_model'].unique()):
            subset = data[data['fc_model'] == fc]
            acc = subset['correct'].mean() * 100
            std = subset.groupby('seed')['correct'].mean().std() * 100
            err = subset['is_error'].mean() * 100
            print(f"  {name:30s} fc={fc}: acc={acc:5.1f}% (std={std:.1f}%)  err={err:5.1f}%  n={len(subset)}")

    # Also compare on BFCL tests (non multi-turn)
    print("\n--- Preserved thinking impact on BFCL (non multi-turn) ---")
    pt_bfcl = df[(df['backend'].str.contains('burrito-pt')) & (df['test_type'] != 'GPT-OSS') & (df['test_name'] != 'multi_turn_base') & (df['reasoning_effort'] == 'medium')]
    b_bfcl = df[(df['backend'].str.contains('burrito@')) & (~df['backend'].str.contains('burrito-pt')) & (df['test_type'] != 'GPT-OSS') & (df['test_name'] != 'multi_turn_base') & (df['reasoning_effort'] == 'medium')]

    for backend_pt in ['burrito-pt@llamacpp', 'burrito-pt@vllm']:
        backend_b = backend_pt.replace('burrito-pt', 'burrito')
        pt_data = pt_bfcl[pt_bfcl['backend'] == backend_pt]
        b_data = b_bfcl[b_bfcl['backend'] == backend_b]
        for test in sorted(pt_data['test_name'].unique()):
            pt_acc = pt_data[pt_data['test_name'] == test]['correct'].mean() * 100
            b_acc = b_data[b_data['test_name'] == test]['correct'].mean() * 100
            print(f"  {backend_pt:25s} vs {backend_b:20s} | {test:20s}: pt={pt_acc:5.1f}%  b={b_acc:5.1f}%  delta={pt_acc-b_acc:+.1f}%")

def section_6_reasoning_effort(df):
    """Not All Reasoning is Created Equal -- effort sweep analysis."""
    print("\n" + "=" * 70)
    print("SECTION 6: NOT ALL REASONING IS CREATED EQUAL")
    print("=" * 70)

    # Burrito backends, responses API, all efforts
    burrito = df[(df['backend'].str.contains('burrito@')) & (~df['backend'].str.contains('burrito-pt')) & (df['wire_api'] == 'responses')]

    # AIME25
    print("\n--- AIME25 by reasoning effort ---")
    aime = burrito[burrito['test_name'] == 'AIME25']
    for effort in ['low', 'medium', 'high']:
        for backend in sorted(aime[aime['reasoning_effort'] == effort]['backend'].unique()):
            subset = aime[(aime['reasoning_effort'] == effort) & (aime['backend'] == backend)]
            for fc in sorted(subset['fc_model'].unique()):
                s = subset[subset['fc_model'] == fc]
                acc = s['correct'].mean() * 100
                std = s.groupby('seed')['correct'].mean().std() * 100
                avg_tok = s['output_token_count'].median()
                avg_reason = s['reasoning_token_count'].median()
                print(f"  {backend:25s} {effort:6s} fc={fc}: acc={acc:5.1f}% (std={std:.1f}%)  out_tok={avg_tok:.0f}  reason_tok={avg_reason:.0f}")

    # GPQA
    print("\n--- GPQA by reasoning effort ---")
    gpqa = burrito[burrito['test_name'] == 'GPQA']
    for effort in ['low', 'medium', 'high']:
        for backend in sorted(gpqa[gpqa['reasoning_effort'] == effort]['backend'].unique()):
            subset = gpqa[(gpqa['reasoning_effort'] == effort) & (gpqa['backend'] == backend)]
            for fc in sorted(subset['fc_model'].unique()):
                s = subset[subset['fc_model'] == fc]
                acc = s['correct'].mean() * 100
                std = s.groupby('seed')['correct'].mean().std() * 100
                avg_tok = s['output_token_count'].median()
                avg_reason = s['reasoning_token_count'].median()
                print(f"  {backend:25s} {effort:6s} fc={fc}: acc={acc:5.1f}% (std={std:.1f}%)  out_tok={avg_tok:.0f}  reason_tok={avg_reason:.0f}")

    # Multi-turn base
    print("\n--- multi_turn_base by reasoning effort (burrito@llamacpp) ---")
    mt = burrito[burrito['test_name'] == 'multi_turn_base']
    for effort in ['low', 'medium', 'high']:
        subset = mt[(mt['reasoning_effort'] == effort) & (mt['backend'] == 'burrito@llamacpp')]
        for fc in sorted(subset['fc_model'].unique()):
            s = subset[subset['fc_model'] == fc]
            acc = s['correct'].mean() * 100
            std = s.groupby('seed')['correct'].mean().std() * 100
            avg_tok = s['output_token_count'].median()
            print(f"  {effort:6s} fc={fc}: acc={acc:5.1f}% (std={std:.1f}%)  out_tok={avg_tok:.0f}")

    # BFCL pooled
    print("\n--- BFCL pooled by reasoning effort (burrito@llamacpp) ---")
    bfcl = burrito[(burrito['test_type'] != 'GPT-OSS') & (burrito['test_name'] != 'multi_turn_base')]
    for effort in ['low', 'medium', 'high']:
        subset = bfcl[(bfcl['reasoning_effort'] == effort) & (bfcl['backend'] == 'burrito@llamacpp')]
        for fc in sorted(subset['fc_model'].unique()):
            s = subset[subset['fc_model'] == fc]
            acc = s['correct'].mean() * 100
            avg_tok = s['output_token_count'].median()
            print(f"  {effort:6s} fc={fc}: acc={acc:5.1f}%  out_tok={avg_tok:.0f}  n={len(s)}")

def section_7_python_tools(df):
    """Python Tool Impact -- AIME25 with python_enabled=1."""
    print("\n" + "=" * 70)
    print("SECTION 7: PYTHON TOOL IMPACT")
    print("=" * 70)

    # Phase 4: AIME25 with python enabled
    py_on = df[(df['python_enabled'] == 1) & (df['test_name'] == 'AIME25')]
    py_off = df[(df['python_enabled'] == 0) & (df['test_name'] == 'AIME25') & (df['backend'].str.contains('burrito@')) & (df['wire_api'] == 'responses')]

    print(f"\nPython ON rows: {len(py_on)}")
    print(f"Python OFF (burrito, responses) rows: {len(py_off)}")

    print("\n--- AIME25 accuracy: python ON vs OFF ---")
    for effort in ['low', 'medium', 'high']:
        py_on_e = py_on[py_on['reasoning_effort'] == effort]
        py_off_e = py_off[py_off['reasoning_effort'] == effort]
        if len(py_on_e) > 0:
            for backend in sorted(py_on_e['backend'].unique()):
                bk_on = py_on_e[py_on_e['backend'] == backend]
                bk_off = py_off_e[py_off_e['backend'] == backend]
                on_acc = bk_on['correct'].mean() * 100
                off_acc = bk_off['correct'].mean() * 100
                on_tok = bk_on['output_token_count'].median()
                off_tok = bk_off['output_token_count'].median()
                on_npy = bk_on['n_tool_calls_python'].sum()
                print(f"  {effort:6s} {backend:25s}: ON={on_acc:5.1f}% (tok={on_tok:.0f}, py_calls={on_npy})  OFF={off_acc:5.1f}% (tok={off_tok:.0f})  delta={on_acc-off_acc:+.1f}%")

def main():
    df = load()
    print(f"Loaded {len(df)} rows")
    print()

    section_1_irrelevance(df)
    section_2_jinja_fix(df)
    section_3_wire_api(df)
    section_4_multi_turn(df)
    section_5_preserved_thinking(df)
    section_6_reasoning_effort(df)
    section_7_python_tools(df)

if __name__ == '__main__':
    main()
