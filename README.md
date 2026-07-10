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

Official Review of Submission11628 by Reviewer Kcgk
Official Reviewby Reviewer Kcgk05 Jul 2026 at 23:05 (modified: 09 Jul 2026 at 01:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Ethics Reviewers, Ethics Chairs, Reviewer Kcgk, AuthorsRevisions
Paper Summary:
The paper proposed to extract named entities from document and assign them more privacy budget than the other tokens. It compares this method with the method where all tokens receives the same privacy budget.

Summary Of Strengths:
The paper identifies a gap in prior word-level text sanitization methods. Many embedding-based sanitizers provide only metric-LDP, where the privacy loss can scale with the diameter of the embedding space, rather than giving a clean full-vocabulary ε-LDP guarantee. DB-San directly addresses this by clipping embedding distances to a public interval, which makes the sensitivity bounded and allows the exponential mechanism to recover pure ε-LDP over the full vocabulary.
The method is conceptually simple and easy to implement. DB-San keeps the familiar word-level exponential mechanism framework, but adds two changes: it clips the embedding distance and assigns different privacy budgets to sensitive and non-sensitive tokens.
Experiments show a meaningful empirical privacy–utility improvement over the main SanText* baseline.
Summary Of Weaknesses:
The empirical comparison is narrow. The paper mainly compares DB-San against SanText*, a bounded-distance version of SanText. That is a clean ablation, but it is not a strong state-of-the-art comparison.

The definition of “sensitive token” depends heavily on NER quality. In the experiments, sensitive tokens are mainly person, organization, and location entities detected by a public NER tagger. This is simple and practical, but many truly sensitive words may not be named entities.

Comments Suggestions And Typos:
NA

Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.
Excitement: 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Limitations And Societal Impact:
Yes

Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
Add:

Author-Editor Confidential Comment

Official Comment
Official Review of Submission11628 by Reviewer mYMg
Official Reviewby Reviewer mYMg04 Jul 2026 at 10:11 (modified: 09 Jul 2026 at 01:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Ethics Reviewers, Ethics Chairs, Reviewer mYMg, AuthorsRevisions
Paper Summary:
This paper introduces DB-San (Dual-Budget Sanitization), a novel word-level text sanitization framework designed to satisfy pure -local differential privacy (-LDP) over a full vocabulary. Prior word-level privatization methods typically offer only metric-LDP (MLDP), where privacy loss scales directly with the unbounded embedding diameter, or restrict indistinguishability to small, disjoint token buckets.

DB-San resolves this dilemma through a two-fold mechanism:Vocabulary Partitioning and Distance Clipping. The framework also introduces a per-document budget redistribution rule to optimize utility by transferring slack from sensitive to non-sensitive tokens. Evaluated on the SST-2 and QNLI benchmarks using GloVe embeddings and a fine-tuned BERT-base classifier, DB-San significantly reduces sensitive token leakage (cutting sensitive-token overlap by  to ) while retaining downstream classifier accuracy within a few percentage points of uniform-budget baselines.

Summary Of Strengths:
Rigorous Privacy Formulation: The paper successfully bridges a prominent gap in word-level local differential privacy. Converting the weaker metric-LDP guarantee into a strict, pure -LDP bound over the entire vocabulary provides a substantial theoretical upgrade over prior works like SanText.

Effective Optimization via Redistribution: The per-document budget redistribution rule (
 
) is highly practical. It leverages entity sparsity in natural language to dynamically minimize noise on safe tokens without exceeding the document-level privacy allocation.

Comprehensive Multi-Dimensional Evaluation: Rather than focusing solely on classification accuracy, the authors evaluate the mechanism across an array of textual distribution and preservation metrics, including BERTScore, Jaccard similarity, MAUVE, and specific sensitive-token overlap weights.

Summary Of Weaknesses:
Critical Citation Hallucination Error: In the bibliography and on line 161, the paper cites: "Justus Mattern, Zhijing Jin, Benjamin Möller, Bernhard Schölkopf, and Mrinmaya Sachan. 2022." The limits of word level differential privacy. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 867-881. The actual author list for that specific NAACL 2022 paper is Justus Mattern, Benjamin Weggenmann, and Florian Kerschbaum (see https://aclanthology.org/2022.findings-naacl.65/).

Vocabulary Constraints and Out-of-Vocabulary (OOV) Vulnerabilities: The mechanism assumes a closed, predefined working vocabulary intersecting GloVe embeddings with tokens from the training corpus. In real-world decentralized LDP execution, a user's input frequently contains typos, neologisms, or domain-specific slang. The paper lacks a concrete strategy or formal analysis regarding how OOV tokens are sanitized or mapped without leaking sensitive context or violating the full-vocabulary indistinguishability bound.

Reliance on Static Embeddings: While the clipping theory is sound, the experimental execution relies entirely on static GloVe-840B embeddings. Modern text classification and generation architectures operate almost exclusively on dense, contextualized representations from Transformers. Bounding sensitivity and calculating meaningful clip limits () inside dynamic, context-dependent token distributions remains unproven here, narrowing the current practical scope of the findings.

Absence of Human-Centric Quality Evaluation: The automated utility metrics (BERTScore, Jaccard, MAUVE) capture semantic alignment well, but word-level exponential substitution frequently yields syntactically disjoint phrases (e.g., the examples in Table 4: "actually's a charming defense seldom affecting journey"). A localized human evaluation or an automated grammaticality/fluency check would clarify whether the output text remains usable for human-in-the-loop applications.

Potential Vulnerabilities to Side-Channel Categorization Bias: While the NER model and thresholds are public, the performance of the dual-budget distribution relies on how clean the semantic boundary behaves. If certain rare or highly specific entities are misclassified as non-sensitive due to out-of-domain alignment, they will absorb significantly less noise (), potentially introducing an unintended re-identification exposure window.

Comments Suggestions And Typos:
Critical Citation Hallucination Error: In the bibliography and on line 161, the paper cites: "Justus Mattern, Zhijing Jin, Benjamin Möller, Bernhard Schölkopf, and Mrinmaya Sachan. 2022." The limits of word level differential privacy. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 867-881. The actual author list for that specific NAACL 2022 paper is Justus Mattern, Benjamin Weggenmann, and Florian Kerschbaum (see https://aclanthology.org/2022.findings-naacl.65/).

Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There exist hallucinations in citations. Further ethical review is necessary.

Needs Ethics Review: Yes
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits
Add:

Author-Editor Confidential Comment

Official Comment
Official Review of Submission11628 by Reviewer WvVc
Official Reviewby Reviewer WvVc04 Jul 2026 at 09:28 (modified: 09 Jul 2026 at 01:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Ethics Reviewers, Ethics Chairs, Reviewer WvVc, AuthorsRevisions
Paper Summary:
This paper proposes DB-San, a word-level text sanitization mechanism for local differential privacy. The method partitions the vocabulary into sensitive and non-sensitive tokens, assigns a lower privacy budget to sensitive tokens, and uses clipped embedding distances in an exponential mechanism to obtain pure ε-LDP over the full vocabulary.

Summary Of Strengths:
The paper addresses an important problem: providing stronger formal privacy guarantees for word-level text sanitization under local differential privacy.
The use of clipped embedding distances is simple but effective. It makes the sensitivity of the exponential mechanism bounded and gives a clearer ε-LDP interpretation than raw metric-LDP mechanisms.
The authors are careful in several places to distinguish between the worst-case formal guarantee and the empirical benefit of the dual-budget design. In particular, the paper correctly notes that the per-token guarantee remains εn-LDP rather than becoming tighter for sensitive tokens.
Summary Of Weaknesses:
The novelty of the core mechanism is somewhat limited. Clipping a utility/distance function to obtain bounded sensitivity is a standard DP technique, and the dual-budget mechanism ultimately has the same worst-case per-token εn-LDP guarantee as a uniform-budget baseline. The main gain is empirical rather than formal, so the paper should be more cautious in presenting DB-San as a fundamentally stronger privacy mechanism.
The construction of the vocabulary and sensitive-token partition raises privacy-accounting concerns. The paper states that the partition should be public, but the experiments derive the working vocabulary from the training splits and run NER on training texts. If the training corpus is private, then both the vocabulary and the sensitive/non-sensitive partition may leak information unless they are constructed from an external public corpus or accounted for in the privacy budget.
The per-document redistribution rule is not fully convincing. Equation (3) changes εn based on the number of sensitive and non-sensitive tokens in a document. These counts are themselves document-dependent and potentially sensitive. The paper acknowledges partition leakage, but a more formal treatment is needed to justify when this redistribution is safe and what exactly remains protected.
There are some technical inconsistencies in the presentation of the privacy analysis. For example, the paper states that cosine distance has range [0, 2] but lists Δ = 1, whereas earlier Δ is defined as h − ℓ. This discrepancy directly affects the calibration of the exponential mechanism and should be fixed. The cross-vocabulary proof also appears to mix normalized and unnormalized distance notation, which makes the stated εn + L bound harder to verify.
The empirical comparison is too narrow. The main experiments compare mostly against SanText*, a modified bounded-distance version of SanText, but do not implement stronger recent baselines such as TEM, CusText, DYNTEXT, CluSanT, or 1-Diffractor under comparable settings. As a result, the experiments isolate the effect of the dual-budget design, but they do not establish that DB-San is competitive with the current state of the art.
The evaluation is limited to SST-2 and QNLI. These are useful benchmarks, but they are not especially representative of privacy-sensitive text domains such as medical notes, legal documents, emails, or user-generated conversational data. Since the sensitive vocabulary is based on NER, evaluation on entity-heavy and domain-specific datasets would be important.
Sensitive-token overlap is a useful diagnostic but not a complete privacy evaluation. Replacing “Nolan” with “Hartley,” for example, may reduce exact overlap while still revealing that a person name occurred and may preserve semantic or contextual clues. The paper would be stronger with reconstruction attacks, entity-inference attacks, or linkage-style evaluations.
Comments Suggestions And Typos:
The title and abstract are clear, but the paper sometimes overstates the contribution by saying that DB-San “recovers pure ε-LDP” without immediately clarifying that the practical worst-case bound is εn and that the dual-budget benefit is empirical.
The paper should more clearly separate three settings: pure per-token εn-LDP, per-class accounting with category leakage, and redistribution with document-dependent counts. These are currently presented together, which may confuse readers.
The statement that vocabulary expansion acts as a “privacy amplifier” should be softened unless accompanied by a formal amplification result. The experiments show lower overlap, but this is not the same as formal privacy amplification.
The discussion of MAUVE is not very useful for QNLI, where the scores are near zero throughout. The authors may consider replacing or supplementing it with more interpretable text quality metrics.
Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.
Excitement: 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
Overall Assessment: 2 = Resubmit next cycle: I think this paper needs substantial revisions that can be completed by the next ARR cycle.
Ethical Concerns:
There are no concerns with this submission

Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 4 = Useful: I would recommend the new datasets to other researchers or developers for their ongoing work.
Software: 4 = Useful: I would recommend the new software to other researchers or developers for their ongoing work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
Add:

Author-Editor Confidential Comment

Official Comment
Official Review of Submission11628 by Reviewer WjDS
Official Reviewby Reviewer WjDS29 Jun 2026 at 10:18 (modified: 09 Jul 2026 at 01:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Ethics Reviewers, Ethics Chairs, Reviewer WjDS, AuthorsRevisions
Paper Summary:
In this work, the authors address a key limitation in word-level differentially private text sanitation, namely in the uniform privacy budget allocation among all word components of a text, regardless of privacy sensitivity. In response, the authors propose a dual budget setup, which retains the worst-case privacy guarantees of previous approaches, but additionally lowers the budget (stricter privatization) on sensitive named entities, thereby targeting empirical privacy gains. In a series of experiments and ablations, the authors clearly demonstrate the value of such an approach, which more effectively masks out sensitive tokens while generally maintaining better utility than previous approaches.

Summary Of Strengths:
I enjoyed reading this paper, and the ideas introduced in the work to augment existing work on word-level DP text sanitization is both useful and sensible. The work builds upon a growing idea in the literature that not all parts of a text are equally as sensitive, so they should not be handled as such in DP text sanitization. This work leverages this idea and transforms it into a simple and intuitive, yet clearly effective approach for more practical DP-based text privatization.

The authors motivate their work well in prior literature, and they correctly point out limitations in the “one size fits all” approaches in terms of privacy budget distribution. I also applaud the authors for extrapolating word-level mechanisms to the document-level, a necessary step for bringing word-level guarantees out of isolation. This and the other formalizations in the work are done well, and also in a readable manner.

Despite some weaknesses named below, the experiments in the work are generally strong, particularly in the inclusion of privacy and utility measures, as well as an analysis of the trade-offs between these. The multiple ablation studies are also a strong point, which are thoughtful and demonstrate the improvements brought by DB-San.

I suggest reading and addressing the weaknesses listed below; otherwise, I believe this is a strong work, and with some revision, could represent a relevant contribution to the field of DP NLP.

Summary Of Weaknesses:
This work seems to ignore some of the more recent advances in this particular sub-field of word-level DP sanitization, notably with the method of Awon et al. from NAACL 2025 [1] (which is only later in the work briefly mentioned), as well as the considerations brought up by Tong et. al. [2], also from NAACL 2025 – the latter is more relevant for evaluation (and along the same lines of Pang et al.). In addition, the authors should also consider previous work on privacy budget allocation for DP text privatization [3], which aligns well with the per-document budget distribution pursued in this work.
The authors should be careful about how they introduce DP-Prompt and LLM-based privatization – these works do not rewrite whole documents under a single budget, but instead leverage per-token privacy budgets (i.e., during sampling). DP-BART, however, is correctly introduced.
The authors should justify why they choose this specific NER model, which seemingly rests on one unpublished preprint. Why were newer/other models not considered?
Why is only SanText included in the comparative evaluations? It seems that CusText should be included as well, considering that the pure-DP vs MLDP comparison is already being made. On this note, the other word-level methods mentioned in the related work could have also been included for a more comprehensive evaluation.
The exact choice of clipping bounds for the distance scores is not clear to me. Why these particular max/mins?
In reporting the number of named entities detected per dataset, it should be made clear what “density ratio” means, although I assumed this means the ratio of non-named entities to named entities.
While the utility metrics are strong and diverse, I feel the empirical privacy evaluation, which solely takes the form of measuring sensitive token overlap, is quite weak. This score is almost self-fulfilling, in the sense that it measures how well the sensitive tokens are masked, which is an inherent component to DB-San (but not to SanText). In addition, these tokens are by design assigned stricter privacy budgets in DB-San, which means they are more likely to be perturbed away from the original tokens. More robust privacy evaluations are needed for higher confidence in the privacy-preserving capabilities of DB-San. For this, refer to the aforementioned Tong et al., as well as more traditional attacks, such as the attackers of Mattern et al. (referenced in the paper).
The analysis of trade-offs mainly takes the form of Figure 3 – it would have been useful to define and calculate trade-off scores that incorporate all of the utilized metrics, allowing for a more direct, empirical comparison.
I realize space is a limiting factor, but the lack of a formal discussion is a missed opportunity to discuss the implications of the findings, particularly in the merits and potential limitations of the dual-budget approach.
[1] Ahmed Musa Awon, Yun Lu, Shera Potka, and Alex Thomo. 2025. CluSanT: Differentially Private and Semantically Coherent Text Sanitization. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3676–3693, Albuquerque, New Mexico. Association for Computational Linguistics.

[2] Meng Tong, Kejiang Chen, Xiaojian Yuan, Jiayang Liu, Weiming Zhang, Nenghai Yu, and Jie Zhang. 2025. On the Vulnerability of Text Sanitization. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5150–5164, Albuquerque, New Mexico. Association for Computational Linguistics.

[3] Stephen Meisenbacher, Chaeeun Joy Lee, and Florian Matthes. 2025. Spend Your Budget Wisely: Towards an Intelligent Distribution of the Privacy Budget in Differentially Private Text Rewriting. In Proceedings of the Fifteenth ACM Conference on Data and Application Security and Privacy (CODASPY '25). Association for Computing Machinery, New York, NY, USA, 84–95.

Comments Suggestions And Typos:
The references contain two works where the arXiv preprint versions are cited, but where the published versions should be cited.
Confidence: 5 = Positive that my evaluation is correct. I read the paper very carefully and am familiar with related work.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 4 = Exciting: I would mention this paper to others and/or make an effort to attend its presentation in a conference.
Overall Assessment: 3.5 = Borderline Conference
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 1 = No usable datasets submitted.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review
Add:

Author-Editor Confidential Comment

Official Comment
Official Review of Submission11628 by Reviewer SeFY
Official Reviewby Reviewer SeFY27 Jun 2026 at 00:47 (modified: 09 Jul 2026 at 01:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Ethics Reviewers, Ethics Chairs, Reviewer SeFY, AuthorsRevisions
Paper Summary:
The paper considers the problem of privatizing/sanitizing text at a word-level using (local) differential privacy. Past work had weaknesses such as privacy guarantees that scaled with the diameter of the embedding set, satisfying weaker notions of differential privacy (e.g. restricting the pairs of adjacent examples), or not adapting the privacy budget to the amount of sensitive content in each example. The authors propose Dual Budget Sanitization (DBSan), which runs an exponential mechanism token-by-token where the utility is the distance to the embedding of the true token. While this approach has been used before, the authors make two modification: First, tokens are partitioned into sensitive and non-sensitive classes using some public information, and then tokens in the non-sensitive class have a higher  used in the exponential mechanism to randomize them. Second, the distance in the embedding space is clipped to an interval  so that a sensitivity bound  holds regardless of the diameter of the embedding space. The authors propose two variants of DBSan, one that normalizes the weights in the exponential mechanism so each token remains within its class w.p. p and is randomized to the opposite class w.p. 1-p, and one that only allows tokens to be randomized to their own class. The authors run experiments comparing to SanText, a prior work. They show with sensitive-token overlap being the metric for privacy, and performance on the downstream task being a metric for utility, that DB-San achieves a better privacy-utility tradeoff than SanText. Ablations show how different instantiations of DBSan and its variants affect various metrics of interest.

Summary Of Strengths:
Removes limitations of past works, e.g. achieves finite epsilon which some past works don't.
Clear presentation and thorough experimentation makes the results and arguments in the paper easy to understand.
Method is sound and exploits a dimension previous works didn't, namely the sensitivity gap between different classes of tokens.
Summary Of Weaknesses:
The authors claim an improvement is satisfying local DP rather than metric-local DP. However, the diameter of the embedding space is dataset-agnostic if we know the whole vocabulary in advance (or for something like a cosine distance, has an upper and lower bound built-in and is easy to calculate), so metric-local DP maps to local DP with some appropriate parameter anyway. Clipping is basically just a way to sidestep this calculation, at the cost of adding an additional hyperparameter, so I think this is barely a limitation of past work. I think mostly what is interesting about clipping is that it 'flattens' the distribution of the exponential mechanism which could maybe optimize the privacy-utility tradeoff, or deal with e.g. an embedding with a few outliers that blows up the diameter, but this doesn't seem to be the motivation of the authors for using clipping.
The analysis the authors use that lets them leverage the per-class privacy budget assumes that releasing for each token whether it is sensitive or not is public information. It's not clear or discussed why this is a reasonable threat model. For example, suppose we know a hospital sends a templated email to anyone who has cancer that we wish to sanitize, and the way we define the sensitive/non-sensitive classes accidentally puts a few names in the non-sensitive class. If we see the sensitive/non-sensitive mask for an example whose sensitive pattern seems to match the hospital's templated email, we can be pretty sure it is an email telling someone they have cancer. Even worse, suppose the name of the recipient is typically 2 sensitive tokens surrounded by non-sensitive tokens at the start of the email, but someone has a uniquely long name that maps to many more sensitive tokens; we could easily tell if this person received an email saying they have cancer using this information.
It's not clear to me why, given we are assuming each token's class is public information, the three variants in Theorem 3 have different privacy guarantees. Note that DP only requires indistinguishability for pairs of examples which have the same public information; otherwise the output of the mechanism doesn't provide any advantage since the adversary can perfectly distinguish the examples using the public information already. For the class-restricted mechanism, the support mismatch only occurs for pairs of examples which have different public information; pairs with the same public information have the same output support. So the class-restricted mechanism retrieves the -DP guarantee as well.  should also retrieve the same guarantee, because the sampling of  is independent of the example, i.e. we could release it publicly and reduce to exponential mechanisms with the same support for pairs ofe xamples with the same public information. I think this means the empirical comparisons between the methods needs to be redone to account for the better accounting.
Figure 3 seems to suggest the given method doesn't actually outperform SanText in certain settings, but the authors don't discuss the figure at all.
Comments Suggestions And Typos:
N/A

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.
Overall Assessment: 2.5 = Borderline Findings
Best Paper Justification:
N/A

Limitations And Societal Impact:
Yes

Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No
Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Knowledge Of Paper Source: N/A, I do not know anything about the paper from outside sources
Impact Of Knowledge Of Paper: N/A, I do not know anything about the paper from outside sources
Reviewer Certification: I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review

