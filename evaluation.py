"""Privacy and utility evaluation metrics.

Metric selection rationale (matching related work):

PRIVACY (lower = better protection):
- Author ID F1:     Gold standard for document-level DP. Used by DP-Prompt
                    and Spend Your Budget. Train classifier on clean, test on
                    sanitized. Lower F1 = harder to identify author.
- ROUGE-1/L:        Textual overlap between original and sanitized.
                    Used by DP-GTR. Lower = less leakage.
- Entity Overlap:   Fraction of original NER entities surviving sanitization.
                    DP-Sketch-specific: validates entity protection works.

UTILITY (higher = better preservation):
- Sentiment Acc:    Classification accuracy on sanitized text.
                    Used by DP-Prompt. Higher = sentiment preserved.
- BERTScore F1:     Semantic similarity (model-based). Standard NLG metric.
- Semantic Sim:     Sentence-transformers cosine. Quick proxy for meaning.
"""

import numpy as np
import spacy


# ======================== PRIVACY METRICS ========================

def compute_rouge(original, sanitized):
    """ROUGE-1 and ROUGE-L (privacy: lower = more private). Used by DP-GTR."""
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r = s.score(original, sanitized)
    return {"rouge1": r["rouge1"].fmeasure, "rougeL": r["rougeL"].fmeasure}


def compute_bleu(original, sanitized):
    """Sentence BLEU (privacy: lower = more private). Used by DP-GTR."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    ref, hyp = original.lower().split(), sanitized.lower().split()
    if not hyp:
        return 0.0
    return sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1)


def compute_entity_overlap(original, sanitized):
    """Fraction of original NER entities surviving in sanitized text.
    DP-Sketch-specific: validates entity replacement works.
    """
    nlp = spacy.load("en_core_web_sm")
    orig_ents = {e.text.lower() for e in nlp(original).ents}
    if not orig_ents:
        return 0.0
    return sum(1 for e in orig_ents if e in sanitized.lower()) / len(orig_ents)


def compute_author_id_f1(clean_texts, clean_labels, sanitized_texts, sanitized_labels):
    """Authorship attribution F1 (privacy: lower = better).
    Used by DP-Prompt, Spend Your Budget Wisely.

    Static attacker: train on clean embeddings, test on sanitized embeddings.
    Returns macro F1 score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Static attacker: trains on clean, evaluates on sanitized
    train_embs = model.encode(clean_texts)
    test_embs = model.encode(sanitized_texts)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_embs, clean_labels)
    preds = clf.predict(test_embs)
    return f1_score(sanitized_labels, preds, average="macro")


# ======================== UTILITY METRICS ========================

def compute_bertscore_batch(originals, sanitized_list):
    """BERTScore F1 for a batch (utility: higher = better)."""
    from bert_score import score as bscore
    _, _, f1 = bscore(sanitized_list, originals, lang="en", verbose=False,
                      rescale_with_baseline=True)
    return f1.tolist()


def compute_sentiment_accuracy(original_labels, sanitized_texts):
    """Sentiment classification accuracy on sanitized text.
    Used by DP-Prompt. Train on clean -> predict on sanitized.
    Higher = better utility (sentiment preserved).
    """
    from sketch import extract_sentiment
    predicted = [extract_sentiment(t) for t in sanitized_texts]
    correct = sum(1 for pred, true in zip(predicted, original_labels) if pred == true)
    return correct / len(original_labels) if original_labels else 0.0


# ======================== COMBINED ========================

def evaluate_pair(original, sanitized, total_eps, similarity_fn=None,
                  orig_sentiment="", san_sentiment=""):
    """Evaluate a single original-sanitized pair."""
    rouge = compute_rouge(original, sanitized)
    return {
        "rouge1": rouge["rouge1"],
        "rougeL": rouge["rougeL"],
        "bleu": compute_bleu(original, sanitized),
        "entity_overlap": compute_entity_overlap(original, sanitized),
        "semantic_sim": similarity_fn(original, sanitized) if similarity_fn else 0.0,
        "sentiment_preserved": orig_sentiment == san_sentiment,
        "total_epsilon": total_eps,
    }


def aggregate(results):
    """Aggregate per-sample results into mean +/- std."""
    if not results:
        return {}
    def ms(key):
        v = np.array([r[key] for r in results], dtype=float)
        return float(v.mean()), float(v.std())
    agg = {"n_samples": len(results)}
    for k in ["rouge1", "rougeL", "bleu", "entity_overlap", "bertscore_f1", "semantic_sim"]:
        if k in results[0]:
            m, s = ms(k)
            agg[k] = m
            agg[f"{k}_std"] = s
    agg["sentiment_acc"] = float(np.mean([r["sentiment_preserved"] for r in results]))
    agg["total_epsilon"] = results[0].get("total_epsilon", 0)
    return agg

