#!/usr/bin/env python3
"""Side-by-side comparison: DP-Sketch vs DP-Prompt vs DP-GTR (simplified).

Shows original, sanitized text from each method, and epsilon spent.

Usage:
    python compare_methods.py --dataset yelp --n 5
"""

import argparse, json, os, re, textwrap
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data import load_data
from models import Generator, Similarity
from sketch import (
    extract_sketch, protect_sketch, reconstruct, extract_sentiment,
    _clean_text, _spacy, extract_keywords, _paraphrase_single,
)
from dp_mechanisms import joint_exponential_mechanism


# ======================== DP-Prompt Baseline ========================

def run_dp_prompt(text, generate_fn, temperature=1.0, sensitivity=10.0):
    """DP-Prompt: zero-shot paraphrase with temperature as DP mechanism.

    eps_per_token = 2 * sensitivity / T
    eps_total = n_tokens * eps_per_token
    """
    cleaned = _clean_text(text)
    prompt = f"Paraphrase the following text:\n\n{cleaned[:600]}\n\nParaphrase:"
    sanitized = generate_fn(prompt, temperature=temperature).strip()
    n_tokens = len(_spacy()(cleaned))
    eps_total = n_tokens * (2 * sensitivity / temperature)
    return sanitized, eps_total


# ======================== DP-GTR Simplified ========================

def run_dp_gtr_simplified(text, generate_fn, m=5, temperature=1.0,
                          sensitivity=10.0, eps_kw=2.0, top_k_kw=5):
    """Simplified DP-GTR: m paraphrases + keyword extraction + ICL reconstruction.

    Stage 1: Generate m paraphrases via DP-Prompt mechanism
    Stage 2: Extract consensus keywords via Joint-EM + pick lowest-perplexity paraphrase
    Stage 3: ICL reconstruction prompt

    eps_total = m * n * (2*sens/T) + eps_kw
    """
    cleaned = _clean_text(text)
    n_tokens = len(_spacy()(cleaned))
    eps_per_token = 2 * sensitivity / temperature

    # Stage 1: Generate m paraphrases
    paraphrases = []
    for _ in range(m):
        prompt = f"Paraphrase the following text:\n\n{cleaned[:600]}\n\nParaphrase:"
        para = generate_fn(prompt, temperature=temperature).strip()
        if len(para) > 10:
            paraphrases.append(para)
        else:
            paraphrases.append(cleaned[:200])

    eps_stage1 = m * n_tokens * eps_per_token

    # Stage 2: Consensus keywords (simplified -- count word freq across paraphrases)
    word_counts = {}
    for para in paraphrases:
        words = set(para.lower().split())
        for w in words:
            if len(w) > 3:  # skip short words
                word_counts[w] = word_counts.get(w, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    kw_scores = np.array([c for _, c in sorted_words[:30]]) if sorted_words else np.array([1.0])
    rng = np.random.default_rng(42)
    actual_k = min(top_k_kw, len(kw_scores))
    if actual_k > 0 and len(kw_scores) > 0:
        sel_idx = joint_exponential_mechanism(kw_scores, actual_k, eps_kw, 1.0, rng)
        released_kw = [sorted_words[i][0] for i in sel_idx]
    else:
        released_kw = []

    # Pick shortest paraphrase as reference (proxy for lowest perplexity)
    reference = min(paraphrases, key=len) if paraphrases else cleaned[:200]

    # Stage 3: ICL reconstruction
    kw_str = ", ".join(released_kw) if released_kw else "general"
    icl_prompt = (
        f"Refer to the following text to generate a new version:\n"
        f"{reference}\n"
        f"Avoid using these tokens: {kw_str}\n"
        f"Generated text:\n"
    )
    sanitized = generate_fn(icl_prompt).strip()

    eps_total = eps_stage1 + eps_kw
    return sanitized, eps_total, released_kw


# ======================== Main Comparison ========================

def wrap(text, width=75, prefix="    "):
    return "\n".join(prefix + line for line in textwrap.wrap(text, width))


def run_comparison(args):
    print(f"Loading models...")
    gen = Generator(args.llm, max_tokens=args.max_tokens)
    sim = Similarity(args.sim_model)

    samples = load_data(args.dataset, args.n, args.seed)
    print(f"Loaded {len(samples)} samples from {args.dataset}\n")

    all_results = []

    for i, s in enumerate(tqdm(samples, desc="Processing")):
        text = s["text"]

        # --- DP-Sketch ---
        raw = extract_sketch(text, args.k_facts)
        protected = protect_sketch(
            raw, generate_fn=gen, similarity_fn=sim, seed=args.seed + i,
        )
        sketch_san = reconstruct(protected, gen)
        sketch_eps = protected["total_epsilon"]

        # --- DP-Prompt (T=1.0) ---
        prompt_san, prompt_eps = run_dp_prompt(text, gen, temperature=1.0)

        # --- DP-GTR simplified (m=3 to save time) ---
        gtr_san, gtr_eps, gtr_kw = run_dp_gtr_simplified(text, gen, m=3, temperature=1.0)

        result = {
            "idx": i, "label": s["label_name"],
            "n_tokens": raw["n_tokens"],
            "original": text,
            "sketch_sanitized": sketch_san, "sketch_eps": sketch_eps,
            "prompt_sanitized": prompt_san, "prompt_eps": prompt_eps,
            "gtr_sanitized": gtr_san, "gtr_eps": gtr_eps,
        }
        all_results.append(result)

        # Print side-by-side
        print(f"\n{'='*80}")
        print(f"  SAMPLE {i+1}/{args.n}  |  label: {s['label_name']}  |  {raw['n_tokens']} tokens")
        print(f"{'='*80}")

        print(f"\n  ORIGINAL:")
        print(wrap(text))

        print(f"\n  DP-Sketch (eps={sketch_eps:.1f}):")
        print(wrap(sketch_san))

        print(f"\n  DP-Prompt T=1 (eps={prompt_eps:.0f}):")
        print(wrap(prompt_san))

        print(f"\n  DP-GTR m=3 (eps={gtr_eps:.0f}):")
        print(wrap(gtr_san))

        # Quick metrics
        orig_words = set(text.lower().split())
        for name, san in [("Sketch", sketch_san), ("Prompt", prompt_san), ("GTR", gtr_san)]:
            san_words = set(san.lower().split())
            overlap = len(orig_words & san_words) / max(len(orig_words), 1)
            sent = extract_sentiment(san)
            print(f"    {name:>8}: overlap={overlap:.0%}  sentiment={sent}  len={len(san.split())}w")

    # Save
    os.makedirs("results", exist_ok=True)
    out = Path("results") / f"comparison_{args.dataset}_n{args.n}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  EPSILON SUMMARY ({args.dataset}, {args.n} samples)")
    print(f"{'='*80}")
    print(f"  {'#':>3} {'Tokens':>6} | {'Sketch':>10} | {'DP-Prompt':>10} | {'DP-GTR':>10} | {'Prompt/Sketch':>14} {'GTR/Sketch':>12}")
    print(f"  {'-'*75}")
    for r in all_results:
        ratio_p = r["prompt_eps"] / max(r["sketch_eps"], 0.01)
        ratio_g = r["gtr_eps"] / max(r["sketch_eps"], 0.01)
        print(f"  {r['idx']:>3} {r['n_tokens']:>6} | {r['sketch_eps']:>10.1f} | {r['prompt_eps']:>10.0f} | {r['gtr_eps']:>10.0f} | {ratio_p:>13.0f}x {ratio_g:>11.0f}x")

    means = {
        "sketch": np.mean([r["sketch_eps"] for r in all_results]),
        "prompt": np.mean([r["prompt_eps"] for r in all_results]),
        "gtr": np.mean([r["gtr_eps"] for r in all_results]),
    }
    print(f"\n  Mean:  Sketch={means['sketch']:.1f}  Prompt={means['prompt']:.0f}  GTR={means['gtr']:.0f}")
    print(f"  Sketch is {means['prompt']/means['sketch']:.0f}x cheaper than DP-Prompt")
    print(f"  Sketch is {means['gtr']/means['sketch']:.0f}x cheaper than DP-GTR")
    print(f"\n  Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yelp")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--sim_model", default="all-MiniLM-L6-v2")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--k_facts", type=int, default=5)
    args = p.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
