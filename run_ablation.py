"""Ablation studies for DP-Sketch."""

import argparse, json, os
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

# Default config (medium budget)
DEFAULTS = dict(k_facts=5, n_cands=5, top_k_kw=5,
                eps_ent=1.0, eps_sent=0.5, eps_kw=2.0, eps_fact=1.5, eps_struct=0.5)

ABLATIONS = {
    "k_facts":      {"param": "k_facts",  "values": [1, 3, 5, 7, 10]},
    "n_cands":      {"param": "n_cands",  "values": [3, 5, 10, 15]},
    "epsilon_fact":  {"param": "eps_fact", "values": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]},
    "top_k_kw":     {"param": "top_k_kw", "values": [3, 5, 7, 10]},
    "components":   {"type": "component_removal"},
}


def _run_config(samples, gen, sim, params, seed):
    """Run DP-Sketch with given params, return aggregate metrics."""
    originals, sanitized_list, evals = [], [], []
    for i, s in enumerate(tqdm(samples, desc="  running", leave=False)):
        r = run_dp_sketch(s["text"], gen, sim, seed=seed + i, **params)
        originals.append(r["original"])
        sanitized_list.append(r["sanitized"])
        ev = evaluate_pair(
            r["original"], r["sanitized"],
            r["protected_sketch"]["total_epsilon"],
            similarity_fn=sim,
            orig_sentiment=s["label_name"],
            san_sentiment=extract_sentiment(r["sanitized"]),
        )
        evals.append(ev)
    bs = compute_bertscore_batch(originals, sanitized_list)
    for ev, b in zip(evals, bs):
        ev["bertscore_f1"] = b
    return aggregate(evals)


def run_sweep(ablation_name, samples, gen, sim, seed):
    """Run a parameter sweep ablation."""
    spec = ABLATIONS[ablation_name]
    param = spec["param"]
    results = {}
    for val in spec["values"]:
        label = f"{param}={val}"
        print(f"  {label}")
        params = {**DEFAULTS, param: val}
        agg = _run_config(samples, gen, sim, params, seed)
        results[label] = agg
        if HAS_WANDB and wandb.run:
            wandb.log({f"ablation/{ablation_name}/{k}": v
                       for k, v in agg.items() if isinstance(v, (int, float))})
    return results


def run_component_removal(samples, gen, sim, seed):
    """Ablation: remove one component at a time."""
    results = {}
    # Full model
    print("  full")
    results["full"] = _run_config(samples, gen, sim, DEFAULTS, seed)

    removals = [
        ("no_entities",  {"eps_ent": 0.0}),
        ("no_sentiment", {"eps_sent": 0.0}),
        ("no_keywords",  {"eps_kw": 0.0}),
        ("no_facts",     {"eps_fact": 0.0}),
        ("no_structure", {"eps_struct": 0.0}),
    ]
    for name, overrides in removals:
        print(f"  {name}")
        params = {**DEFAULTS, **overrides}
        results[name] = _run_config(samples, gen, sim, params, seed)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yelp")
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm", default="google/flan-t5-base")
    p.add_argument("--sim_model", default="all-MiniLM-L6-v2")
    p.add_argument("--ablation", default="k_facts",
                   choices=list(ABLATIONS.keys()) + ["all"])
    p.add_argument("--output_dir", default="results")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", default="dp-sketch")
    args = p.parse_args()

    if args.use_wandb and HAS_WANDB:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"ablation_{args.ablation}_{args.dataset}")

    gen = Generator(args.llm)
    sim = Similarity(args.sim_model)
    samples = load_data(args.dataset, args.n_samples, args.seed)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    to_run = list(ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]
    all_results = {}

    for ab_name in to_run:
        print(f"\n--- Ablation: {ab_name} ---")
        if ABLATIONS[ab_name].get("type") == "component_removal":
            all_results[ab_name] = run_component_removal(samples, gen, sim, args.seed)
        else:
            all_results[ab_name] = run_sweep(ab_name, samples, gen, sim, args.seed)

    # Print summary
    for ab_name, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"  {ab_name}")
        print(f"  {'Config':<20} {'Eps':>6} {'R1':>7} {'BERTSc':>7} {'SemSim':>7} {'EntOvl':>7}")
        print(f"  {'-'*62}")
        for cfg, m in results.items():
            print(f"  {cfg:<20} {m.get('total_epsilon',0):>6.1f} "
                  f"{m['rouge1']:>7.4f} {m.get('bertscore_f1',0):>7.4f} "
                  f"{m['semantic_sim']:>7.4f} {m['entity_overlap']:>7.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = Path(args.output_dir) / f"ablation_{args.ablation}_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
