# DP-Sketch

Code for **"DP-Sketch: Differentially Private Document Sanitization via Sketch Extraction and LLM Reconstruction"**.

## Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
uv run python -m spacy download en_core_web_sm
```

## Run

```bash
# Preliminary (30 samples, fast)
uv run python run_sketch.py --dataset yelp --n_samples 30

# Baseline comparison
uv run python run_baseline.py --dataset yelp --n_samples 30

# Full experiment with wandb
uv run python run_sketch.py --dataset yelp --n_samples 200 --use_wandb

# Ablations
uv run python run_ablation.py --dataset yelp --n_samples 50 --ablation k_facts
uv run python run_ablation.py --dataset yelp --n_samples 50 --ablation epsilon_fact
uv run python run_ablation.py --dataset yelp --n_samples 50 --ablation components
uv run python run_ablation.py --dataset yelp --n_samples 50 --ablation all
```

## Structure

```
dp_mechanisms.py   # Pure DP primitives (EM, Joint-EM, Laplace, RR)
sketch.py          # Extract -> Protect -> Reconstruct pipeline
evaluation.py      # Privacy + utility metrics
data.py            # Dataset loaders (Yelp, IMDb, AG News, SST-2)
models.py          # Flan-T5 generator + sentence-transformers similarity
run_sketch.py      # Main experiment entry point
run_baseline.py    # DP-Prompt baseline
run_ablation.py    # Ablation studies
```
