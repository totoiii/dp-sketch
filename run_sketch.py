"""Main DP-Sketch experiment."""

import argparse, json, time, os
from pathlib import Path
from tqdm import tqdm

from data import load_data
from models import Generator, Similarity
from sketch import run_dp_sketch, extract_sentiment
from evaluation import evaluate_pair, compute_bertscore_batch, aggregate

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    p = argparse.ArgumentParser(description="DP-Sketch experiment")
    p.add_argument("--dataset", default="yelp", choices=["yelp", "imdb", "ag_news", "csqa"])
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--sim_model", default="all-MiniLM-L6-v2")
    p.add_argument("--max_tokens", type=int, default=256)
    # Sketch params
    p.add_argument("--k_facts", type=int, default=5)
    p.add_argument("--n_cands", type=int, default=5)
    p.add_argument("--top_k_kw", type=int, default=5)
    # Epsilon budgets
    p.add_argument("--eps_ent", type=float, default=1.0)
    p.add_argument("--eps_sent", type=float, default=0.5)
    p.add_argument("--eps_kw", type=float, default=2.0)
    p.add_argument("--eps_fact", type=float, default=1.5)
    p.add_argument("--eps_struct", type=float, default=0.5)
    # Output
    p.add_argument("--output_dir", default="results")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="dp-sketch")
    return p.parse_args()


def main():
    args = parse_args()

    # wandb
    if args.use_wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"sketch_{args.dataset}_n{args.n_samples}_k{args.k_facts}")
    elif args.use_wandb:
        print("WARNING: wandb not installed, skipping")

    # Load models
    print(f"Loading LLM: {args.llm}")
    gen = Generator(args.llm, args.max_tokens)
    print(f"Loading similarity: {args.sim_model}")
    sim = Similarity(args.sim_model)

    # Load data
    print(f"Loading {args.dataset} (n={args.n_samples})")
    samples = load_data(args.dataset, args.n_samples, args.seed)
    print(f"Loaded {len(samples)} samples")

    # Run
    originals, sanitized_list, evals = [], [], []
    t0 = time.time()

    for i, s in enumerate(tqdm(samples, desc="DP-Sketch")):
        result = run_dp_sketch(
            s["text"], gen, sim,
            k_facts=args.k_facts, n_cands=args.n_cands, top_k_kw=args.top_k_kw,
            eps_ent=args.eps_ent, eps_sent=args.eps_sent, eps_kw=args.eps_kw,
            eps_fact=args.eps_fact, eps_struct=args.eps_struct,
            seed=args.seed + i,
        )
        originals.append(result["original"])
        sanitized_list.append(result["sanitized"])

        ev = evaluate_pair(
            result["original"], result["sanitized"],
            result["protected_sketch"]["total_epsilon"],
            similarity_fn=sim,
            orig_sentiment=s["label_name"],
            san_sentiment=extract_sentiment(result["sanitized"]),
        )
        evals.append(ev)

        if args.use_wandb and HAS_WANDB:
            wandb.log({"sample_idx": i, **{f"sample/{k}": v for k, v in ev.items()
                                           if isinstance(v, (int, float))}})

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # BERTScore batch
    print("Computing BERTScore...")
    bs = compute_bertscore_batch(originals, sanitized_list)
    for ev, b in zip(evals, bs):
        ev["bertscore_f1"] = b

    agg = aggregate(evals)
    agg["time_seconds"] = elapsed

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / f"sketch_{args.dataset}_k{args.k_facts}_n{args.n_samples}.json"
    report = {"config": vars(args), "aggregate": agg,
              "per_sample": [{"orig": o[:200], "san": s[:200], **e}
                             for o, s, e in zip(originals, sanitized_list, evals)]}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Log to wandb
    if args.use_wandb and HAS_WANDB:
        wandb.summary.update({f"agg/{k}": v for k, v in agg.items() if isinstance(v, (int, float))})
        wandb.log(agg)

    # Print
    print(f"\n{'='*50}")
    print(f"DP-Sketch | {args.dataset} | eps={agg['total_epsilon']:.1f}")
    print(f"{'='*50}")
    print(f"  ROUGE-1:       {agg['rouge1']:.4f} +/- {agg['rouge1_std']:.4f}")
    print(f"  Entity Overlap:{agg['entity_overlap']:.4f} +/- {agg['entity_overlap_std']:.4f}")
    print(f"  BERTScore F1:  {agg['bertscore_f1']:.4f} +/- {agg['bertscore_f1_std']:.4f}")
    print(f"  Semantic Sim:  {agg['semantic_sim']:.4f} +/- {agg['semantic_sim_std']:.4f}")
    print(f"  Sentiment Acc: {agg['sentiment_acc']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
