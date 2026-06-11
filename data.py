"""Dataset loaders.

Dataset selection rationale (matching related work):
- Yelp:  Used by DP-Prompt (Utpala 2023) and Spend Your Budget (Meisenbacher 2025).
         Long reviews with author labels -> authorship attribution attack.
- IMDb:  Used by DP-Prompt. Long movie reviews with sentiment labels.
         Same evaluation protocol as our main baseline.
- AG News: Optional. Not used by related work directly, but useful for
           domain generalization experiments (news vs reviews).

- CSQA: Used by DP-GTR. 5-choice QA. Tests whether sanitized text still
         answers questions correctly. Adds task diversity beyond sentiment.
- DocVQA: Used by DP-GTR. Pre-extracted OCR tokens + open-answer QA.
          Dataset provides OCR words directly (no image processing needed).
- MedQA: Used by DP-GTR. Medical QA (USMLE-style, 4-choice).
         Tests domain transfer to specialized medical text.

NOT included:
- SST-2/QNLI: Too short for document-level rewriting (~20 words avg).
              Used by word-level methods (SanText, CusText, DB-San).
"""

from datasets import load_dataset


def _cap(text, max_len=800):
    return text[:max_len]


def load_yelp(n=100, seed=42):
    """Yelp review polarity. Binary sentiment.
    Used by: DP-Prompt, Spend Your Budget Wisely.
    """
    ds = load_dataset("fancyzhx/yelp_polarity", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    labels = {0: "negative", 1: "positive"}
    return [{"text": _cap(r["text"]), "label": r["label"],
             "label_name": labels[r["label"]], "dataset": "yelp"} for r in ds]


def load_imdb(n=100, seed=42):
    """IMDb movie reviews. Binary sentiment.
    Used by: DP-Prompt.
    """
    ds = load_dataset("stanfordnlp/imdb", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    labels = {0: "negative", 1: "positive"}
    return [{"text": _cap(r["text"]), "label": r["label"],
             "label_name": labels[r["label"]], "dataset": "imdb"} for r in ds]


def load_ag_news(n=100, seed=42):
    """AG News topic classification. 4 classes.
    Optional: domain generalization (news vs reviews).
    """
    ds = load_dataset("fancyzhx/ag_news", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    labels = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
    return [{"text": _cap(r["text"]), "label": r["label"],
             "label_name": labels[r["label"]], "dataset": "ag_news"} for r in ds]


def load_csqa(n=100, seed=42):
    """CommonsenseQA. 5-choice closed-answer QA.
    Used by: DP-GTR. Tests if sanitized prompt still yields correct answer.
    """
    ds = load_dataset("tau/commonsense_qa", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    results = []
    for r in ds:
        # Build the full prompt as DP-GTR does: question + choices
        choices = " ".join(f"({l}) {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
        text = f"{r['question']} {choices}"
        results.append({"text": text, "label": r["answerKey"],
                        "label_name": r["answerKey"], "dataset": "csqa"})
    return results


def load_docvqa(n=100, seed=42):
    """PFL-DocVQA. Open-answer QA over pre-extracted OCR tokens.
    Used by: DP-GTR. The 'document' is a list of OCR words from a form/receipt.
    Utility = ROUGE-1 between LLM answer and ground truth.
    """
    ds = load_dataset("nielsr/docvqa_1200_examples_donut", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    results = []
    for r in ds:
        words = r.get("words", [])
        if not words:
            continue
        question = r["query"]["en"] if isinstance(r["query"], dict) else str(r["query"])
        text = f"Extracted OCR tokens: {', '.join(words)}\nQuestion: {question}"
        answers = r.get("answers", [])
        label = answers[0] if answers else ""
        results.append({"text": _cap(text, 1200), "label": label,
                        "label_name": label, "dataset": "docvqa"})
    return results


def load_medqa(n=100, seed=42):
    """MedQA USMLE 4-option. Medical board-style QA.
    Used by: DP-GTR. Tests domain transfer to specialized medical text.
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    results = []
    for r in ds:
        options = r["options"]
        choices = " ".join(f"({k}) {v}" for k, v in options.items())
        text = f"{r['question']} {choices}"
        results.append({"text": _cap(text), "label": r["answer_idx"],
                        "label_name": r["answer_idx"], "dataset": "medqa"})
    return results


LOADERS = {
    "yelp": load_yelp, "imdb": load_imdb, "ag_news": load_ag_news,
    "csqa": load_csqa, "docvqa": load_docvqa, "medqa": load_medqa,
}


def load_data(name, n=100, seed=42):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(LOADERS)}")
    return LOADERS[name](n, seed)
