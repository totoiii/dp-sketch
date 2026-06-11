# DP-Sketch

Code for **"DP-Sketch: Differentially Private Document Sanitization via Sketch Extraction and LLM Reconstruction"**.

DP-Sketch operates in three phases:
1. **Extract** a compact sketch (entities, sentiment, keywords, key facts, structure) from the input.
2. **Protect** each sketch component with a calibrated DP mechanism (Exponential, Joint-EM, Randomized Response, Laplace).
3. **Reconstruct** a new text from the protected sketch using an LLM.

Total privacy budget scales as O(K+E) instead of O(n) (DP-Prompt) or O(m×n) (DP-GTR).

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd dp-sketch
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
python -m spacy download en_core_web_sm
```

## Download datasets

First run requires internet to cache datasets from HuggingFace:

```bash
python -c "
from datasets import load_dataset
for name, split in [
    ('fancyzhx/yelp_polarity', 'test'),
    ('tau/commonsense_qa', 'validation'),
    ('nielsr/docvqa_1200_examples_donut', 'test'),
    ('GBaker/MedQA-USMLE-4-options', 'test'),
]:
    load_dataset(name, split=split)
    print(f'OK: {name}')
"
```

After this, all subsequent runs work offline with `HF_HUB_OFFLINE=1`.

## Quick smoke test

Run on 1 sample to verify the pipeline works end-to-end:

```bash
python sanity_check.py --dataset yelp --n 1
```

## Side-by-side comparison (DP-Sketch vs DP-Prompt vs DP-GTR)

```bash
# Single dataset, 5 samples
python compare_methods.py --dataset yelp --n 5

# All 4 datasets, 10 samples each (~15 min)
for ds in yelp csqa docvqa medqa; do
    python compare_methods.py --dataset $ds --n 10
done

# Epsilon-only comparison (no text output, faster)
python compare_epsilon.py --dataset yelp --n 10
```

Results are saved to `results/`.

## Full experiments

```bash
# DP-Sketch main experiment
python run_sketch.py --dataset yelp --n_samples 30

# DP-Prompt baseline at multiple temperatures
python run_baseline.py --dataset yelp --n_samples 30

# Ablation studies
python run_ablation.py --dataset yelp --n_samples 50 --ablation k_facts
python run_ablation.py --dataset yelp --n_samples 50 --ablation epsilon_fact
python run_ablation.py --dataset yelp --n_samples 50 --ablation components
python run_ablation.py --dataset yelp --n_samples 50 --ablation all

# With wandb logging
python run_sketch.py --dataset yelp --n_samples 200 --use_wandb
```

Available datasets: `yelp`, `imdb`, `ag_news`, `csqa`, `docvqa`, `medqa`.

## Project structure

```
sketch.py            # Core pipeline: Extract -> Protect -> Reconstruct
dp_mechanisms.py     # Pure DP primitives (EM, Joint-EM, Laplace, RR)
models.py            # Flan-T5 generator + sentence-transformers similarity
data.py              # Dataset loaders (Yelp, IMDb, AG News, CSQA, DocVQA, MedQA)
evaluation.py        # Privacy + utility metrics (ROUGE, BERTScore, overlap)
sanity_check.py      # Quick smoke test: 1-3 samples with full sketch printout
compare_methods.py   # Side-by-side: DP-Sketch vs DP-Prompt vs DP-GTR
compare_epsilon.py   # Epsilon budget comparison across methods
run_sketch.py        # Main DP-Sketch experiment
run_baseline.py      # DP-Prompt baseline experiment
run_ablation.py      # Ablation studies (k_facts, epsilon, components)
```
