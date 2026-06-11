"""DP-Prompt baseline: zero-shot LLM paraphrasing with temperature."""

import argparse, json, time, os
from pathlib import Path
from tqdm import tqdm

from data import load_data
from models import Generator, Similarity
from sketch import extract_sentiment
from evaluation import evaluate_pair, compute_bertscore_batch, aggregate

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    p = argparse.ArgumentParser(description="DP-Prompt baseline")
    p.add_argument("--dataset", default="yelp", choices=["yelp", "imdb", "ag_news", "csqa"])
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--sim_model", default="all-MiniLM-L6-v2")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperatures", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    p.add_argument("--output_dir", default="results")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="dp-sketch")
    return p.parse_args()


def main():
    args = parse_args()

    if args.use_wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"baseline_{args.dataset}_n{args.n_samples}")

    gen = Generator(args.llm, args.max_tokens)
    sim = Similarity(args.sim_model)
    samples = load_data(args.dataset, args.n_samples, args.seed)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    all_results = {}
    for temp in args.temperatures:
        print(f"\n--- DP-Prompt T={temp} ---")
        originals, sanitized_list, evals = [], [], []
        t0 = time.time()

        for s in tqdm(samples, desc=f"T={temp}"):
            prompt = f"Paraphrase the following text:\n\n{s['text'][:800]}\n\nParaphrase:"
            san = gen(prompt, temperature=temp)
            originals.append(s["text"])
            sanitized_list.append(san)

            # Approximate DP-Prompt epsilon: n_tokens * sensitivity / T
            n_tok = len(s["text"].split())
            approx_eps = 2 * n_tok * 10 / temp  # simplified approximation

            ev = evaluate_pair(
                s["text"], san, approx_eps,
                similarity_fn=sim,
                orig_sentiment=s["label_name"],
                san_sentiment=extract_sentiment(san),
            )
            evals.append(ev)

        elapsed = time.time() - t0
        bs = compute_bertscore_batch(originals, sanitized_list)
        for ev, b in zip(evals, bs):
            ev["bertscore_f1"] = b

        agg = aggregate(evals)
        agg["temperature"] = temp
        agg["time_seconds"] = elapsed
        all_results[f"T={temp}"] = agg

        if args.use_wandb and HAS_WANDB:
            wandb.log({f"baseline_T{temp}/{k}": v for k, v in agg.items()
                       if isinstance(v, (int, float))})

        print(f"  ROUGE-1: {agg['rouge1']:.4f}  EntOvlp: {agg['entity_overlap']:.4f}  "
              f"BERTSc: {agg['bertscore_f1']:.4f}  SemSim: {agg['semantic_sim']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = Path(args.output_dir) / f"baseline_{args.dataset}_n{args.n_samples}.json"
    with open(out_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
