#!/usr/bin/env python3
"""Sanity check: run DP-Sketch on 3 samples and show original vs sanitized.

Usage:
    python sanity_check.py
    python sanity_check.py --dataset imdb --n 5
"""

import argparse, textwrap

from data import load_data
from models import Generator, Similarity
from sketch import extract_sketch, protect_sketch, reconstruct


def wrap(text, width=80, prefix="  "):
    return "\n".join(prefix + line for line in textwrap.wrap(text, width))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yelp")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--k_facts", type=int, default=5)
    p.add_argument("--eps_fact", type=float, default=1.5)
    args = p.parse_args()

    print("Loading models...")
    gen = Generator(args.llm)
    sim = Similarity()

    print(f"Loading {args.n} samples from {args.dataset}...")
    samples = load_data(args.dataset, args.n, seed=42)

    for i, s in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{args.n}  |  label: {s['label_name']}")
        print(f"{'='*80}")

        print(f"\n--- ORIGINAL ---")
        print(wrap(s["text"]))

        # Phase 1: Extract
        raw = extract_sketch(s["text"], args.k_facts, gen)
        print(f"\n--- EXTRACTED SKETCH (no DP cost) ---")
        print(f"  Sentiment:  {raw['sentiment']}")
        print(f"  Entities:   {raw['entities']}")
        print(f"  Keywords:   {raw['keywords'][0][:5]}")
        print(f"  Facts:")
        for j, fact in enumerate(raw["facts"]):
            print(f"    {j+1}. {fact}")
        print(f"  Tokens: {raw['n_tokens']}  Sentences: {raw['n_sents']}")

        # Phase 2: Protect
        protected = protect_sketch(
            raw, eps_fact=args.eps_fact, generate_fn=gen, similarity_fn=sim, seed=42+i,
        )
        print(f"\n--- PROTECTED SKETCH (eps={protected['total_epsilon']:.2f}) ---")
        print(f"  Sentiment:  {protected['sentiment']}")
        print(f"  Entities:   {protected['entities']}")
        print(f"  Keywords:   {protected['keywords']}")
        print(f"  Facts:")
        for j, fact in enumerate(protected["facts"]):
            print(f"    {j+1}. {fact}")
        print(f"  Budget:     {protected['budget']}")

        # Phase 3: Reconstruct
        sanitized = reconstruct(protected, gen)
        print(f"\n--- SANITIZED OUTPUT ---")
        print(wrap(sanitized))

        print(f"\n--- QUICK COMPARISON ---")
        orig_words = set(s["text"].lower().split())
        san_words = set(sanitized.lower().split())
        overlap = len(orig_words & san_words) / max(len(orig_words), 1)
        print(f"  Word overlap: {overlap:.1%}")
        print(f"  Orig length:  {len(s['text'].split())} words")
        print(f"  San length:   {len(sanitized.split())} words")


if __name__ == "__main__":
    main()
