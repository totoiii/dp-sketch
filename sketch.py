"""DP-Sketch: Extract -> Protect -> Reconstruct pipeline.

Key design: fact extraction is EXTRACTIVE (NLP-based sentence scoring),
not generative. This avoids depending on LLM instruction-following for
structured output. The LLM is only used for paraphrasing individual
sentences (simple task) and final reconstruction.
"""

import json
import re

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from dp_mechanisms import (
    exponential_mechanism, joint_exponential_mechanism,
    laplace_mechanism, randomized_response,
)

_NLP = None


def _spacy():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ======================== PHASE 1: EXTRACT (no DP cost) ========================


def extract_entities(text):
    doc = _spacy()(text)
    ents = {}
    for e in doc.ents:
        ents.setdefault(e.label_, [])
        if e.text not in ents[e.label_]:
            ents[e.label_].append(e.text)
    return ents


def extract_sentiment(text):
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        s = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        return "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
    except ImportError:
        return "neutral"


def extract_keywords(text, top_k=10):
    doc = _spacy()(text)
    sents = [s.text for s in doc.sents if len(s.text.strip()) > 5] or [text]
    vec = TfidfVectorizer(stop_words="english", max_features=500)
    try:
        mat = vec.fit_transform(sents)
    except ValueError:
        return [], []
    names = vec.get_feature_names_out()
    scores = mat.mean(axis=0).A1
    top = scores.argsort()[::-1][:top_k]
    return [names[i] for i in top], [float(scores[i]) for i in top]


def _clean_text(text):
    """Normalize whitespace and newlines for proper sentence splitting."""
    text = re.sub(r'\\n', '\n', text)  # literal \n to real newlines
    text = re.sub(r'\n{2,}', '. ', text)  # paragraph breaks -> sentence boundary
    text = re.sub(r'\n', ' ', text)       # remaining newlines -> space
    text = re.sub(r'\s{2,}', ' ', text)   # collapse whitespace
    return text.strip()


def extract_key_sentences(text, k):
    """Extract top-k most informative sentences via TF-IDF scoring.

    Extractive approach: no LLM needed, deterministic, robust.
    Each sentence is scored by its mean TF-IDF weight across the document.
    """
    cleaned = _clean_text(text)
    doc = _spacy()(cleaned)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 15]
    if not sents:
        sents = [cleaned]

    if len(sents) <= k:
        return sents[:k]

    vec = TfidfVectorizer(stop_words="english")
    try:
        mat = vec.fit_transform(sents)
    except ValueError:
        return sents[:k]

    sent_scores = mat.sum(axis=1).A1
    top_indices = sent_scores.argsort()[::-1][:k]
    # Return in original order for coherence
    top_indices_sorted = sorted(top_indices)
    return [sents[i] for i in top_indices_sorted]


def extract_sketch(text, k_facts, generate_fn=None):
    """Extract raw sketch. generate_fn is unused (extractive approach)."""
    cleaned = _clean_text(text)
    doc = _spacy()(cleaned)
    n_sents = len(list(doc.sents))
    # Adapt k to actual content: don't ask for 5 facts from a 2-sentence text
    effective_k = min(k_facts, max(1, n_sents))
    return {
        "text": text,
        "entities": extract_entities(cleaned),
        "sentiment": extract_sentiment(cleaned),
        "keywords": extract_keywords(cleaned),
        "facts": extract_key_sentences(cleaned, effective_k),
        "n_sents": n_sents,
        "n_tokens": len(doc),
    }


# ======================== PHASE 2: PROTECT (DP cost here) ========================

PERSON_POOL = ["Alex Johnson", "Maria Garcia", "James Wilson", "Sarah Chen",
               "Robert Kim", "Emily Davis", "Michael Brown", "Lisa Wang",
               "David Martinez", "Jennifer Lee", "Thomas Anderson", "Anna Schmidt"]
LOC_POOL = ["Springfield", "Riverside", "Oakland", "Portland", "Madison",
            "Georgetown", "Fairview", "Burlington", "Arlington", "Lakewood"]
ORG_POOL = ["Acme Corp", "Global Solutions", "Pacific Industries",
            "Summit Tech", "Atlas Group", "Pinnacle Systems", "Metro Holdings"]
POOLS = {"PERSON": PERSON_POOL, "GPE": LOC_POOL, "LOC": LOC_POOL, "ORG": ORG_POOL}


_PARA_PROMPTS = [
    "Paraphrase the following sentence in your own words:\n{fact}\nParaphrase:",
    "Rewrite this sentence with the same meaning but different wording:\n{fact}\nRewritten:",
    "Express the same idea differently:\n{fact}\nAlternative:",
    "Say this another way:\n{fact}\nAnother way:",
    "Rephrase:\n{fact}\nRephrased:",
]


def _is_garbage(text):
    """Detect garbage output: code-like, repetitive junk, too short."""
    if len(text.split()) < 3:
        return True
    junk_patterns = [r'[a-z]\s*=\s*[\[\'"]', r'[a-z],[a-z],[a-z],[a-z]',
                     r'\b(fact|additional information)\b', r'_{2,}']
    for pat in junk_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _paraphrase_single(fact, generate_fn, attempt=0):
    """Generate a single paraphrase, cycling through prompt templates.

    Retries up to 3 times if the output is garbage or echoes the input.
    """
    prompt_tmpl = _PARA_PROMPTS[attempt % len(_PARA_PROMPTS)]
    prompt = prompt_tmpl.format(fact=fact)
    result = generate_fn(prompt, max_tokens=80).strip()

    if _is_garbage(result):
        if attempt < 3:
            return _paraphrase_single(fact, generate_fn, attempt + 1)
        return fact  # fallback: return original (will be entity-replaced later)

    # Check for echo
    orig_words = set(fact.lower().split())
    res_words = set(result.lower().split())
    overlap = len(orig_words & res_words) / max(len(orig_words), 1)
    if overlap > 0.85 and attempt < 3:
        return _paraphrase_single(fact, generate_fn, attempt + 1)

    return result


def protect_entities(entities, epsilon, rng):
    result, n = {}, 0
    for etype, elist in entities.items():
        pool = POOLS.get(etype)
        if not pool:
            continue
        replaced = []
        for _ in elist:
            idx = exponential_mechanism(np.ones(len(pool)), epsilon, 1.0, rng)
            replaced.append(pool[idx])
            n += 1
        result[etype] = replaced
    return result, epsilon * n


def protect_sentiment(sentiment, epsilon, rng):
    labels = ["positive", "negative", "neutral"]
    idx = labels.index(sentiment) if sentiment in labels else 2
    noisy = randomized_response(idx, 3, epsilon, rng)
    return labels[noisy], epsilon


def protect_keywords(kw, scores, top_k, epsilon, rng):
    if not kw or epsilon <= 0:
        return [], 0.0
    actual_k = min(top_k, len(kw))
    sel = joint_exponential_mechanism(np.array(scores[:len(kw)]), actual_k, epsilon, 1.0, rng)
    return [kw[i] for i in sel], epsilon


def protect_facts(facts, eps_per_fact, n_cands, generate_fn, similarity_fn, rng):
    """Protect each fact: generate M paraphrase candidates, select via EM.

    Skips placeholder/filler facts to avoid wasting budget on garbage.
    """
    _filler = {"no additional details", "additional information", ""}
    real_facts = [f for f in facts if f.strip().lower() not in _filler
                  and not re.match(r'^additional information \(fact \d+\)', f, re.I)]
    if not real_facts:
        return [], 0.0

    protected, total = [], 0.0
    for fact in real_facts:
        candidates = []
        for j in range(n_cands):
            para = _paraphrase_single(fact, generate_fn, attempt=j % len(_PARA_PROMPTS))
            if not _is_garbage(para):
                candidates.append(para)

        # Fallback: if all candidates are garbage, use the original
        if not candidates:
            candidates = [fact]

        sims = np.array([similarity_fn(fact, c) for c in candidates])
        if sims.max() > sims.min():
            sims = (sims - sims.min()) / (sims.max() - sims.min())
        else:
            sims = np.ones_like(sims)

        idx = exponential_mechanism(sims, eps_per_fact, 1.0, rng)
        protected.append(candidates[idx])
        total += eps_per_fact
    return protected, total


def protect_sketch(raw, eps_ent=1.0, eps_sent=0.5, eps_kw=2.0,
                   eps_fact=1.5, eps_struct=0.5, top_k_kw=5, n_cands=5,
                   generate_fn=None, similarity_fn=None, seed=42):
    rng = np.random.default_rng(seed)
    kw, kw_scores = raw["keywords"]

    p_ent, e1 = protect_entities(raw["entities"], eps_ent, rng)
    p_sent, e2 = protect_sentiment(raw["sentiment"], eps_sent, rng)
    p_kw, e3 = protect_keywords(kw, kw_scores, top_k_kw, eps_kw, rng)
    p_facts, e4 = protect_facts(raw["facts"], eps_fact, n_cands, generate_fn, similarity_fn, rng)

    # Replace original entity mentions in facts with protected ones (post-processing, FREE)
    p_facts = _replace_entities_in_facts(p_facts, raw["entities"], p_ent)

    n_s = max(1, round(laplace_mechanism(raw["n_sents"], eps_struct / 2, 1.0, rng)))
    n_t = max(10, round(laplace_mechanism(raw["n_tokens"], eps_struct / 2, 1.0, rng)))
    length = "short" if n_t < 50 else ("medium" if n_t < 150 else "long")
    e5 = eps_struct

    budget = {"entities": e1, "sentiment": e2, "keywords": e3, "facts": e4, "structure": e5}
    return {
        "entities": p_ent, "sentiment": p_sent, "keywords": p_kw,
        "facts": p_facts, "n_sents": n_s, "length": length,
        "budget": budget, "total_epsilon": sum(budget.values()),
    }


def _replace_entities_in_facts(facts, orig_entities, prot_entities):
    """Replace original entity mentions in fact text with protected ones.

    Post-processing on already-DP-protected data, so this is FREE.
    """
    replaced_facts = []
    for fact in facts:
        new_fact = fact
        for etype in orig_entities:
            if etype not in prot_entities:
                # Entity type was dropped (no pool) -- redact
                for orig_name in orig_entities[etype]:
                    new_fact = new_fact.replace(orig_name, "[REDACTED]")
                continue
            orig_list = orig_entities[etype]
            prot_list = prot_entities[etype]
            for i, orig_name in enumerate(orig_list):
                replacement = prot_list[i % len(prot_list)] if prot_list else "[REDACTED]"
                new_fact = new_fact.replace(orig_name, replacement)
        replaced_facts.append(new_fact)
    return replaced_facts


# ======================== PHASE 3: RECONSTRUCT (free post-processing) ========================

_RECON_REVIEW = (
    "Write a {sentiment} {length} review ({n} sentences). "
    "The tone must be clearly {sentiment}.\n\n"
    "Key points to cover:\n{facts}\n\n"
    "Keywords to include: {kw}\n"
    "{ent_section}"
    "Review:\n"
)

_RECON_QA = (
    "Based on the following information, construct a question with answer choices.\n\n"
    "Key information:\n{facts}\n\n"
    "Keywords: {kw}\n"
    "{ent_section}"
    "Question with choices:\n"
)

_RECON_DOC = (
    "Using the following extracted information, write a {length} document summary.\n\n"
    "Key points:\n{facts}\n\n"
    "Keywords: {kw}\n"
    "{ent_section}"
    "Summary:\n"
)


def _detect_domain(sketch):
    """Detect text domain from sketch content for prompt routing."""
    all_text = " ".join(sketch.get("facts", [])).lower()
    kw_text = " ".join(sketch.get("keywords", [])).lower()
    combined = all_text + " " + kw_text
    if any(w in combined for w in ["(a)", "(b)", "(c)", "(d)", "question"]):
        return "qa"
    if any(w in combined for w in ["ocr", "extracted", "tokens:", "document"]):
        return "doc"
    return "review"


def reconstruct(sketch, generate_fn):
    kw = ", ".join(sketch["keywords"]) or "general topics"
    facts = "\n".join(f"- {f}" for f in sketch["facts"] if f.strip())
    if not facts:
        facts = "- General commentary"

    ent_lines = ""
    if sketch["entities"]:
        parts = [f"{t}: {', '.join(ns)}" for t, ns in sketch["entities"].items()]
        ent_lines = "Mention these names/places: " + "; ".join(parts) + "\n"

    domain = _detect_domain(sketch)

    if domain == "qa":
        prompt = _RECON_QA.format(kw=kw, facts=facts, ent_section=ent_lines)
    elif domain == "doc":
        prompt = _RECON_DOC.format(
            length=sketch["length"], kw=kw, facts=facts, ent_section=ent_lines)
    else:
        prompt = _RECON_REVIEW.format(
            sentiment=sketch["sentiment"], length=sketch["length"],
            n=sketch["n_sents"], kw=kw, facts=facts, ent_section=ent_lines)

    # Set minimum output length based on sketch structure
    n_facts = len([f for f in sketch["facts"] if f.strip()])
    min_tokens = max(30, n_facts * 12)  # at least ~12 words per fact

    return generate_fn(prompt, max_tokens=200, min_tokens=min_tokens).strip()


# ======================== FULL PIPELINE ========================


def run_dp_sketch(text, generate_fn, similarity_fn, k_facts=5, n_cands=5,
                  top_k_kw=5, eps_ent=1.0, eps_sent=0.5, eps_kw=2.0,
                  eps_fact=1.5, eps_struct=0.5, seed=42):
    raw = extract_sketch(text, k_facts)
    protected = protect_sketch(
        raw, eps_ent, eps_sent, eps_kw, eps_fact, eps_struct,
        top_k_kw, n_cands, generate_fn, similarity_fn, seed,
    )
    sanitized = reconstruct(protected, generate_fn)
    return {"original": text, "sanitized": sanitized,
            "raw_sketch": raw, "protected_sketch": protected}
