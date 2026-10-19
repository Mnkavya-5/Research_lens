# ResearchLens evaluation (2026-10-16, k=8)

12 answerable + 4 unanswerable questions; labels written by an AI assistant from the papers (see qa_set.json).

## Q&A metrics (percent unless noted)

| Metric | Value |
|---|---|
| Retrieval hit@8 (gold paper in top 8) | 100 |
| Retrieval MRR | 0.85 |
| Evidence recall@8 (answer text retrieved) | 67 |
| Answer correctness | 67 |
| Cites the gold paper (answered questions) | 100 |
| Faithfulness (supported sentences) | 60 |
| Invalid citations | 0 |
| False refusals (answerable but refused) | 8 |
| Correct refusals (unanswerable) | 50 |
| Unanswerable: refused OR warned | 100 |
| Answered questions with a warning | 55 |

## Ranking (48 labelled papers)

| Variant | P@5 | P@10 | nDCG@10 |
|---|---|---|---|
| similarity only | 0.00 | 0.30 | 0.42 |
| weights, raw similarity (current default) | 0.60 | 0.50 | 0.67 |
| weights, min-max relevance (tried, rejected) | 0.20 | 0.30 | 0.50 |

## Per question

| Question | Rank | Evidence | Correct | Refused | Supported | Answer |
|---|---|---|---|---|---|---|
| How many question-answer-reference triples were used to eval | 4 | False | False | True | 0/1 | The collected papers do not contain enough information to answer this. |
| How many items does the ISLAMICFAITHQA benchmark contain? | 1 | True | True | False | 2/3 | The ISLAMICFAITHQA benchmark contains 3,810 Arabic questions and English questions [4]. It is a bilingual generative benchmark designed to measure hallucination |
| Which languages does the ISLAMICFAITHQA benchmark cover? | 1 | False | True | False | 2/3 | The ISLAMICFAITHQA benchmark covers Arabic and English questions [1]. It includes both generative QA and machine reading comprehension formats [3]. The benchmar |
| Which question answering datasets were used in the experimen | 2 | True | True | False | 0/3 | The experimental comparison of naive, enhanced, and agentic RAG used several question-answering datasets. For Question Answering (QA), the datasets included FIQ |
| Which DSPy optimizers did RAGalyst use to refine its metric  | 1 | True | True | False | 1/1 | RAGalyst used the COPRO and MIPROv2 optimizers to refine its metric prompts [2], and also applied the LabeledFewShot optimizer to further optimize the Answer Co |
| Which benchmark was RAGalyst's Answer Correctness metric val | 1 | False | False | False | 0/2 | The Answer Correctness metric in RAGalyst was validated against a human-aligned dataset [1], and the results showed that the A-RAG pipeline outperformed the bas |
| Which benchmark datasets were used to evaluate the privacy a | 1 | True | True | False | 1/1 | The benchmark datasets used to evaluate the privacy agent in the synthetic dataset generation framework are PII-Masking-200K, PWI-Masking-200K, and PHI-Masking- |
| How many HotpotQA questions were used in the component ablat | 2 | True | True | False | 1/1 | The component ablation study of agentic RAG used 5,000 questions from the HotpotQA distractor development set [2]. |
| What exact match score did the full agentic RAG pipeline ach | 1 | False | False | False | 1/3 | The full agentic RAG pipeline achieved an exact match (EM) score of 55.0% on the HotpotQA dataset [4]. This result is presented in the ablation study, where the |
| Which families of hallucination detectors does OpenHalDet su | 1 | True | True | False | 3/3 | OpenHalDet supports black-box, gray-box, and white-box families of hallucination detectors [5]. It integrates representative detectors from these categories int |
| How many comment-edit pairs are in the soup-dataset used by  | 1 | True | True | False | 1/2 | The soup-dataset used by RAG-Reflect contains 5,000 manually annotated comment-edit pairs [7]. These pairs are collected from Java answers with code examples an |
| Which framework coordinates the agents in the agentic AI fra | 1 | True | False | False | 3/3 | The agentic AI framework for 6G RAN automation coordinates the agents using a super agent powered by a Hierarchical Online Decision Transformer (H-ODT) [2]. The |
| What is the boiling point of mercury reported in these paper (unanswerable) | - | - | - | True | - | The collected papers do not contain enough information to answer this. |
| Which paper evaluates retrieval-augmented generation for med (unanswerable) | - | - | - | False | - | The paper evaluated retrieval-augmented generation for medical image segmentation [5], highlighting the importance of reliable retrieval techniques in therapeut |
| By how much did GPT-7 improve accuracy over GPT-6 in these e (unanswerable) | - | - | - | True | - | The collected papers do not contain enough information to answer this. |
| Which dataset of Mars rover telemetry logs was used for eval (unanswerable) | - | - | - | False | - | The evaluation mentioned in the sources refers to the PWI, PHI, and PII datasets, which are synthetic datasets used for privacy evaluation. However, the questio |
