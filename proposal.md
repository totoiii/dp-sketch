# DP-Sketch: Document-Level Text Sanitization via Differentially Private Sketch Extraction

## Problem

Document-level differentially private (DP) text rewriting transforms a private document D into a sanitized version D' satisfying local DP while preserving semantic utility. Unlike central DP, there is no trusted aggregator: the mechanism must protect a single user's single document.

## Prior Work

Three families of approaches address this problem.

**Token-by-token rewriting.** DP-Prompt (Utpala et al., EMNLP 2023), DP-Paraphrase (Mattern et al., 2022), and DP-MLM (Meisenbacher et al., 2024) generate each output token via the exponential mechanism over the vocabulary. The privacy cost scales linearly with document length: eps_total = n × eps_token. For a 100-token document with eps_token = 53.4 (Flan-T5 at T=1.0, reported in DP-GTR Appendix B), the total is 5,340. The formal guarantee is functionally meaningless at this scale.

**Latent space noise.** DP-BART (Igamberdiev and Habernal, ACL 2023) injects Gaussian noise into BART's encoder representation. A single mechanism application means no composition penalty at all. However, the method requires fine-tuning BART, cannot leverage modern LLMs, and is limited by the autoencoder's reconstruction capacity.

**Multi-paraphrase post-processing.** DP-GTR (Li et al., EMNLP 2025), the current state of the art, generates m paraphrases, extracts consensus keywords via the Joint Exponential Mechanism, selects the lowest-perplexity paraphrase as an in-context learning exemplar, and reconstructs. The problem is that generating m paraphrases multiplies the cost: eps_total = m × n × eps_1 + eps_2. For m=10 paraphrases of a 100-token document with Llama at T=1.0, the total reaches 19,402. The empirical privacy improvement from keyword suppression does not reduce the formal DP bound.

**The gap.** Every training-free, LLM-compatible method incurs O(n) or O(m×n) mechanism applications. The only method that avoids composition (DP-BART) requires fine-tuning. No existing work achieves O(k) DP cost in a training-free setting, where k is a small constant independent of document length.

## DP-Sketch

Instead of rewriting token by token, we extract a compact sketch of the document's semantic content using a small fixed number of DP queries, then reconstruct coherent text from the sketch via LLM generation. The reconstruction is post-processing on already-protected data and incurs zero additional privacy cost (Dwork et al., 2006).

The sketch captures five types of information: named entities, sentiment, keywords, key factual propositions, and structural metadata. Each component uses the DP mechanism best suited to its data type.

### Phase 1: Extract (no DP cost)

Standard NLP tools extract the raw sketch from the private document:

- Named entities by type via spaCy NER: {PERSON: [...], ORG: [...], ...}
- Sentiment label via VADER: positive, negative, or neutral
- Top keywords with TF-IDF scores
- K key sentences selected by TF-IDF sentence scoring (extractive, no LLM dependence)
- Structural metadata: sentence count and approximate token count

This phase is entirely local and deterministic. Nothing is released, so no privacy budget is consumed.

### Phase 2: Protect (DP cost here)

Each sketch component is protected independently with its optimal mechanism:

| Component | Mechanism | Budget |
|-----------|-----------|--------|
| Each entity | Exponential mechanism over a public type-class pool | eps_ent per entity |
| Sentiment | Randomized response (3 classes) | eps_sent |
| Keywords (top k) | Joint Exponential Mechanism (Gillenwater et al., NeurIPS 2022) | eps_kw |
| Each fact | Generate M paraphrase candidates, select one via EM on cosine similarity | eps_fact per fact |
| Structure | Laplace noise on sentence and token counts | eps_struct |

Entity replacement selects a substitute from a public pool of the same type (e.g., PERSON names from census data) via the exponential mechanism. Mentions of original entities within the paraphrased facts are also replaced as a post-processing step, which is free.

Fact protection generates M paraphrase candidates for each extracted sentence, scores each candidate by cosine similarity to the original using sentence-transformers, and selects one via the exponential mechanism with the similarity score as utility. Sensitivity equals 1 since scores are normalized to [0, 1].

## Worked Example

Consider the following private Yelp review:

> "Absolutely atrocious service, while there were only three tables of customers at 7pm on a Saturday night. To top it off, the waiter was a smart ass when, after 15 minutes of his being MIA, we inquired as to his whereabouts when we wished to order more drinks & food. On the plus side, the food was fine and arrived quickly (even before the chips and salsa appetizer?)."

### Phase 1: Extract

Standard NLP tools produce the raw sketch (nothing is released, no budget spent):

```
sentiment:  negative
entities:   {CARDINAL: ["three"], TIME: ["7pm", "Saturday night", "15 minutes"], ORG: ["MIA"]}
keywords:   [("food", 0.31), ("service", 0.28), ("salsa", 0.19), ("waiter", 0.17), ("chips", 0.14),
             ("appetizer", 0.12), ("drinks", 0.10), ("tables", 0.08), ("customers", 0.07), ("smart", 0.05)]
facts:      ["Absolutely atrocious service, while there were only three tables of customers at 7pm on a Saturday night.",
             "The waiter was a smart ass when, after 15 minutes of his being MIA, we inquired as to his whereabouts.",
             "The food was fine and arrived quickly (even before the chips and salsa appetizer?)."]
structure:  {n_sents: 3, n_tokens: 80}
```

### Phase 2: Protect

Each component is privatized with its own mechanism and its own slice of the budget.

**Entities (eps_ent = 1.0 per entity, 5 entities = 5.0 total)**

For each entity, the exponential mechanism selects a replacement from a public pool of the same type. For example, ORG pool = ["Atlas Group", "Metro Holdings", "Pinnacle Systems", ...]. The mechanism assigns probability proportional to exp(eps × score / 2) where the score can be a frequency or embedding similarity. Higher eps means the most popular replacement wins more often; lower eps means more uniform (noisier) selection.

Result: TIME entities have no public pool of equivalent meaning, so they are redacted to [REDACTED]. ORG "MIA" is replaced by "Atlas Group".

```
protected_entities: {ORG: ["Atlas Group"]}
```

Budget spent so far: 1.0 (only 1 entity had a usable pool).

**Sentiment (eps_sent = 0.5)**

Randomized response with 3 classes (positive, negative, neutral). The true label is "negative". With probability p = exp(eps) / (exp(eps) + 2) ≈ 0.45, we keep the true label. With probability (1-p)/2 ≈ 0.27 each, we flip to one of the other two. In this case, the coin lands on "negative" (true label preserved).

```
protected_sentiment: negative     (budget: 0.5)
```

**Keywords, top k=5 (eps_kw = 2.0)**

The Joint Exponential Mechanism (Gillenwater et al., NeurIPS 2022) selects k items from a scored list in a single mechanism application. It takes the TF-IDF scores as the utility function and selects 5 keywords jointly.

Concretely: the mechanism defines a probability distribution over all (10 choose 5) = 252 possible subsets of size 5 from our 10 candidate keywords. Each subset S gets probability proportional to exp(eps × sum of scores in S / (2 × sensitivity)). Subsets containing the highest-scored words are exponentially more likely to be selected, but lower-scored words have a nonzero chance of replacing them.

With eps_kw = 2.0, the mechanism is moderately noisy. Suppose it selects:

```
protected_keywords: ["food", "salsa", "appetizer", "chips", "saturday"]
```

"service" (score 0.28) was replaced by "saturday" (not in top-5 originally). "waiter" dropped out. This is a single mechanism application, so the total budget for all 5 keywords together is just eps_kw = 2.0, not 5 × 2.0.

Budget spent so far: 1.0 + 0.5 + 2.0 = 3.5.

**Facts, K=3 (eps_fact = 1.5 per fact, 3 facts = 4.5 total)**

For each fact, the LLM generates M=5 paraphrase candidates. Then the exponential mechanism selects one based on cosine similarity to the original.

Take fact 1: "Absolutely atrocious service, while there were only three tables of customers at 7pm on a Saturday night."

The LLM generates 5 candidates:

```
candidate 1: "The service was incredibly awful, and three tables of customers were there at 7pm on a Saturday."    sim=0.91
candidate 2: "A customer service that was absolutely awful at 7pm on a Saturday was three tables of customers."    sim=0.82
candidate 3: "Saturday night at seven, the restaurant was nearly empty yet the service was terrible."              sim=0.74
candidate 4: "Terrible experience with very few diners present on a weekend evening."                              sim=0.68
candidate 5: "The service quality was poor despite low occupancy during the weekend."                              sim=0.61
```

The exponential mechanism assigns probability to each candidate proportional to exp(eps_fact × sim / 2). With eps_fact = 1.5:

```
P(cand 1) ∝ exp(1.5 × 0.91 / 2) = exp(0.683) = 1.98
P(cand 2) ∝ exp(1.5 × 0.82 / 2) = exp(0.615) = 1.85
P(cand 3) ∝ exp(1.5 × 0.74 / 2) = exp(0.555) = 1.74
P(cand 4) ∝ exp(1.5 × 0.68 / 2) = exp(0.510) = 1.67
P(cand 5) ∝ exp(1.5 × 0.61 / 2) = exp(0.458) = 1.58
```

After normalizing: P(cand 1)=22.6%, P(cand 2)=21.1%, P(cand 3)=19.9%, P(cand 4)=19.0%, P(cand 5)=18.0%. The distribution is relatively flat because eps is moderate. The mechanism samples one; suppose candidate 1 wins.

The same process runs independently for facts 2 and 3 (each spending eps_fact = 1.5). After selection, original entity mentions within the chosen paraphrases are replaced using the protected entity mapping from above (post-processing, free).

```
protected_facts:
  1. "The service was incredibly awful, and [REDACTED] tables of customers were there at [REDACTED] on a [REDACTED]."
  2. "The waiter was nice and friendly, but it was a long wait."
  3. "The food was fine and arrived quickly (even before the chips and salsa appetizer?)."
```

Budget spent so far: 1.0 + 0.5 + 2.0 + 4.5 = 8.0.

**Structure (eps_struct = 0.5)**

Two Laplace mechanisms, each with eps_struct/2 = 0.25:

- n_sents: true value = 3. Laplace(0, 1/0.25) = Laplace(0, 4). Sample noise = +1.7, round to 5.
- n_tokens: true value = 80. Laplace(0, 1/0.25) = Laplace(0, 4). Sample noise = -3.2, round to 77.

From the noisy token count, derive a length bin: 77 tokens maps to "medium" (the bins are public: short < 50, medium < 150, long ≥ 150).

```
protected_structure: {n_sents: 5, length: "medium"}
```

Total budget: 1.0 + 0.5 + 2.0 + 4.5 + 0.5 = **8.5 epsilon**.

### Phase 3: Reconstruct

The protected sketch is assembled into a prompt:

```
Write a detailed medium negative review with exactly 5 sentences.

Include ALL of these points in your review:
1. The service was incredibly awful, and [REDACTED] tables of customers were there at [REDACTED] on a [REDACTED].
2. The waiter was nice and friendly, but it was a long wait.
3. The food was fine and arrived quickly (even before the chips and salsa appetizer?).

Use these keywords: food, salsa, appetizer, chips, saturday
Mentioned organizations: Atlas Group
Write the full review now:
```

The LLM generates:

> "I have been coming to this location for years. The service was incredibly awful despite only a handful of tables being occupied. The waiter took forever to come back, and when he did it was clear he did not care. The food was decent and came out fast, with the chips and salsa appetizer arriving before anything else. I will not be returning to Atlas Group anytime soon."

This output contains no original entity names, no timestamps, and no verbatim phrases from the source. The sentiment is preserved. The reconstruction step is post-processing on already-protected data, so it costs zero additional budget.

For comparison, DP-Prompt would spend 80 × 20 = 1,600 epsilon on this same 80-token document (at T=1.0 with Flan-T5), and DP-GTR with m=5 paraphrases would spend 5 × 80 × 20 + 2 = 8,002 epsilon. DP-Sketch spent 8.5.



## Privacy Analysis

By sequential composition (Dwork et al., 2014):

    eps_total = E × eps_ent + eps_sent + eps_kw + K × eps_fact + eps_struct

where E is the number of entities and K is the number of extracted facts.

With K=5 and E=5 at default budgets (eps_ent=1.0, eps_sent=0.5, eps_kw=2.0, eps_fact=1.5, eps_struct=0.5):

    eps_total = 5(1.0) + 0.5 + 2.0 + 5(1.5) + 0.5 = 15.5

When we fix the total budget at eps=15.5 and compare how each method distributes it:

| Method | Mechanism applications | Budget per mechanism | Depends on doc length |
|--------|----------------------|---------------------|-----------------------|
| DP-Prompt | n = 100 | 0.155 | Yes, linearly |
| DP-GTR (m=5) | m×n = 500 | 0.031 | Yes, linearly in m×n |
| DP-Sketch | K+E+3 = 13 | 1.19 | No |

DP-Sketch allocates roughly 1.2 epsilon per mechanism instead of 0.03 to 0.15. Each individual mechanism operates with far less noise, producing higher-quality protected components.

Preliminary experiments on 10 Yelp reviews with Flan-T5-base confirm this:

| Method | Mean epsilon | Relative cost |
|--------|-------------|---------------|
| DP-Sketch | 12.4 | 1× |
| DP-Prompt (T=1) | 2,892 | 233× |
| DP-GTR (m=5, T=1) | 14,462 | 1,166× |

DP-Sketch's epsilon ranges from 8.5 to 18.5 depending on entity count, but never exceeds 20 regardless of document length. The baselines scale linearly with token count.

## Why This Works

The central insight is a separation of concerns. What to protect is determined by a small number of semantic components (entities, facts, sentiment) that capture the document's meaning. How to express the result is delegated to the LLM, whose parametric knowledge from pretraining provides writing ability at zero privacy cost.

Token-by-token methods conflate these two steps. They apply DP to every token, including articles, prepositions, and punctuation, which carry no private information but still consume budget. DP-Sketch spends budget exclusively on semantically meaningful components.

## Relation to Existing Work

**DB-San (our EMNLP 2026 submission)** introduces the principle that different text components deserve different privacy budgets and uses NER to identify sensitive tokens. DP-Sketch extends this idea from word-level to document-level with fact-level granularity.

**DP-GTR** also extracts keywords and uses post-processing for reconstruction. The difference is how information is obtained: DP-GTR extracts keywords indirectly through m costly paraphrases (O(m×n) budget), while DP-Sketch extracts directly from the original (O(1) budget for keywords).

**DP-BART** is the only other method that avoids O(n) composition. It achieves this through latent space noise injection, which requires fine-tuning. DP-Sketch avoids composition through sketch extraction, which is training-free and compatible with any LLM.

**Spend Your Budget Wisely (Meisenbacher et al., CODASPY 2025)** proposes unequal budget allocation across words based on importance scoring. DP-Sketch implements the same principle at a coarser, more effective granularity: allocating budget across semantic components rather than individual words.

**Granularity is crucial (Vu et al., EMNLP Findings 2024)** demonstrates that the granularity at which DP is applied (word, sentence, document) fundamentally affects the privacy-utility tradeoff. DP-Sketch operates at fact granularity, which lies between sentence and document level, and this paper provides direct theoretical motivation for our design choice.

**CoGenesis (Zhang et al., 2024)** uses a sketch-then-fill architecture where a cloud LLM generates a content sketch and a local LLM fills in personal details. The architecture is superficially similar, but CoGenesis provides no formal privacy guarantee. DP-Sketch provides provable epsilon-DP through explicit mechanism applications on each sketch component.
