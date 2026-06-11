#!/usr/bin/env python3
"""Compare DP-Sketch epsilon vs baselines on all available datasets.

Shows per-sample epsilon breakdown and computes what DP-Prompt and DP-GTR
would cost for the same documents.

Usage:
    python compare_epsilon.py
    python compare_epsilon.py --n 10 --dataset yelp
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import load_data
from models import Generator, Similarity
from sketch import extract_sketch, protect_sketch, reconstruct, extract_sentiment


def dp_prompt_epsilon(n_tokens, temperature=1.0):
    """DP-Prompt: eps = n * (2 * sensitivity / T).

    Following DP-GTR's LLMDP.py: sensitivity = |b_max - b_min| (logit range).
    For Flan-T5-base, typical logit range ~ 10-15. We use 10 (conservative).
    """
    sensitivity = 10.0  # typical logit range for Flan-T5
    eps_per_token = 2 * sensitivity / temperature
    return n_tokens * eps_per_token


def dp_gtr_epsilon(n_tokens, m_paraphrases=5, temperature=1.0, eps_keywords=2.0):
    """DP-GTR: eps = m * n * (2 * sens / T) + eps_keywords.

    Stage 1: m paraphrases, each n tokens via EM.
    Stage 2: Joint-EM for keywords.
    """
    sensitivity = 10.0
    eps_per_token = 2 * sensitivity / temperature
    return m_paraphrases * n_tokens * eps_per_token + eps_keywords


def run_comparison(args):
    print(f"Loading models...")
    gen = Generator(args.llm, max_tokens=args.max_tokens)
    sim = Similarity(args.sim_model)

    datasets_to_test = args.datasets.split(",")
    all_results = {}

    for ds_name in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds_name} ({args.n} samples)")
        print(f"{'='*70}")

        try:
            samples = load_data(ds_name, args.n, args.seed)
        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")
            continue

        results = []
        for i, s in enumerate(tqdm(samples, desc=f"  {ds_name}")):
            text = s["text"]

            # --- DP-Sketch ---
            raw = extract_sketch(text, args.k_facts)
            protected = protect_sketch(
                raw, eps_ent=args.eps_ent, eps_sent=args.eps_sent,
                eps_kw=args.eps_kw, eps_fact=args.eps_fact, eps_struct=args.eps_struct,
                top_k_kw=args.top_k_kw, n_cands=args.n_cands,
                generate_fn=gen, similarity_fn=sim, seed=args.seed + i,
            )
            sanitized = reconstruct(protected, gen)

            n_tokens = raw["n_tokens"]
            n_entities = sum(len(v) for v in raw["entities"].values())
            n_facts = len(raw["facts"])

            # --- Baseline epsilons for SAME document ---
            eps_dp_prompt_t05 = dp_prompt_epsilon(n_tokens, temperature=0.5)
            eps_dp_prompt_t10 = dp_prompt_epsilon(n_tokens, temperature=1.0)
            eps_dp_prompt_t15 = dp_prompt_epsilon(n_tokens, temperature=1.5)
            eps_dp_gtr = dp_gtr_epsilon(n_tokens, m_paraphrases=5, temperature=1.0)

            result = {
                "idx": i,
                "dataset": ds_name,
                "label": s["label_name"],
                "n_tokens": n_tokens,
                "n_entities": n_entities,
                "n_facts": n_facts,
                "n_sents": raw["n_sents"],
                # DP-Sketch breakdown
                "sketch_eps_entities": protected["budget"]["entities"],
                "sketch_eps_sentiment": protected["budget"]["sentiment"],
                "sketch_eps_keywords": protected["budget"]["keywords"],
                "sketch_eps_facts": protected["budget"]["facts"],
                "sketch_eps_structure": protected["budget"]["structure"],
                "sketch_eps_total": protected["total_epsilon"],
                "sketch_n_mechanisms": n_entities + 1 + 1 + n_facts + 1,
                # Baselines
                "dp_prompt_T05": eps_dp_prompt_t05,
                "dp_prompt_T10": eps_dp_prompt_t10,
                "dp_prompt_T15": eps_dp_prompt_t15,
                "dp_gtr_m5_T10": eps_dp_gtr,
                # Output preview
                "original_preview": text[:150],
                "sanitized_preview": sanitized[:150],
                "original_sentiment": s["label_name"],
                "sanitized_sentiment": extract_sentiment(sanitized),
            }
            results.append(result)

        all_results[ds_name] = results

        # Print per-sample table
        print(f"\n  {'#':>3} {'Tok':>4} {'Ent':>3} {'K':>2} | "
              f"{'Sketch eps':>10} {'(breakdown)':>35} | "
              f"{'DP-Prompt T=1':>13} {'DP-GTR m=5':>11} | {'Ratio':>6}")
        print(f"  {'-'*105}")
        for r in results:
            breakdown = (f"ent={r['sketch_eps_entities']:.1f} "
                        f"sent={r['sketch_eps_sentiment']:.1f} "
                        f"kw={r['sketch_eps_keywords']:.1f} "
                        f"fact={r['sketch_eps_facts']:.1f} "
                        f"str={r['sketch_eps_structure']:.1f}")
            ratio = r["dp_prompt_T10"] / max(r["sketch_eps_total"], 0.01)
            print(f"  {r['idx']:>3} {r['n_tokens']:>4} {r['n_entities']:>3} {r['n_facts']:>2} | "
                  f"{r['sketch_eps_total']:>10.1f} {breakdown:>35} | "
                  f"{r['dp_prompt_T10']:>13.0f} {r['dp_gtr_m5_T10']:>11.0f} | {ratio:>5.0f}x")

        # Aggregate
        sketch_eps = [r["sketch_eps_total"] for r in results]
        prompt_eps = [r["dp_prompt_T10"] for r in results]
        gtr_eps = [r["dp_gtr_m5_T10"] for r in results]
        print(f"\n  SUMMARY ({ds_name}):")
        print(f"    DP-Sketch:      mean eps = {np.mean(sketch_eps):>8.1f}  (std={np.std(sketch_eps):.1f})")
        print(f"    DP-Prompt T=1:  mean eps = {np.mean(prompt_eps):>8.0f}  ({np.mean(prompt_eps)/np.mean(sketch_eps):.0f}x worse)")
        print(f"    DP-GTR m=5:     mean eps = {np.mean(gtr_eps):>8.0f}  ({np.mean(gtr_eps)/np.mean(sketch_eps):.0f}x worse)")

    # Save all
    os.makedirs("results", exist_ok=True)
    out_path = Path("results") / "epsilon_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Final cross-dataset summary
    print(f"\n{'='*70}")
    print(f"  CROSS-DATASET EPSILON COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Dataset':<10} {'n':>3} {'Sketch eps':>11} {'DP-Prompt':>11} {'DP-GTR':>11} {'Sketch advantage':>18}")
    print(f"  {'-'*68}")
    for ds_name, results in all_results.items():
        se = np.mean([r["sketch_eps_total"] for r in results])
        pe = np.mean([r["dp_prompt_T10"] for r in results])
        ge = np.mean([r["dp_gtr_m5_T10"] for r in results])
        print(f"  {ds_name:<10} {len(results):>3} {se:>11.1f} {pe:>11.0f} {ge:>11.0f} {pe/se:>11.0f}x vs Prompt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="yelp,imdb")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--sim_model", default="all-MiniLM-L6-v2")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--k_facts", type=int, default=5)
    p.add_argument("--n_cands", type=int, default=5)
    p.add_argument("--top_k_kw", type=int, default=5)
    p.add_argument("--eps_ent", type=float, default=1.0)
    p.add_argument("--eps_sent", type=float, default=0.5)
    p.add_argument("--eps_kw", type=float, default=2.0)
    p.add_argument("--eps_fact", type=float, default=1.5)
    p.add_argument("--eps_struct", type=float, default=0.5)
    args = p.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
