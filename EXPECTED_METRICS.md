# EXPECTED_METRICS.md

This file records target metrics from the DiLM paper for reproduction checks.

Source paper:

- arXiv: <https://arxiv.org/abs/2404.00264>
- ACL Anthology PDF: <https://aclanthology.org/2024.findings-naacl.199.pdf>
- Official code: <https://github.com/arumaekawa/DiLM>

Important: the DiLM paper evaluates `SST-2`, `QQP`, and `MNLI-m`, not AG News. Metrics below are therefore paper targets for DiLM reproduction, not expected metrics for our current AG News notebooks.

Reported metrics:

- `SST-2`: accuracy.
- `QQP`: average of accuracy and F1.
- `MNLI-m`: accuracy.
- Values are mean +/- standard deviation over 100 trained learner models unless stated otherwise.
- `DPC` means data per class.

## Table 1: Same-Model BERT_BASE Evaluation

Learner model: `bert-base-uncased` / BERT_BASE.

| Method | SST-2 DPC=5 | SST-2 DPC=10 | SST-2 DPC=20 | QQP DPC=5 | QQP DPC=10 | QQP DPC=20 | MNLI-m DPC=5 | MNLI-m DPC=10 | MNLI-m DPC=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 58.1 +/- 5.2 | 64.3 +/- 7.4 | 70.3 +/- 6.8 | 51.5 +/- 5.6 | 56.0 +/- 4.8 | 59.1 +/- 3.8 | 35.6 +/- 2.1 | 37.7 +/- 2.6 | 40.1 +/- 3.2 |
| K-centers | 70.8 +/- 4.1 | 75.9 +/- 4.7 | 79.8 +/- 3.5 | 60.7 +/- 3.8 | 60.9 +/- 3.1 | 62.6 +/- 2.7 | 36.2 +/- 2.4 | 41.8 +/- 3.2 | 45.3 +/- 3.0 |
| Herding | 70.2 +/- 5.7 | 73.2 +/- 5.7 | 76.9 +/- 4.4 | 56.0 +/- 5.6 | 59.7 +/- 4.1 | 62.3 +/- 3.4 | 36.2 +/- 3.8 | 38.7 +/- 3.7 | 42.8 +/- 3.5 |
| TDD embed. | 89.6 +/- 0.4 | - | - | 81.5 +/- 0.2 | - | - | 75.6 +/- 0.2 | - | - |
| TDD text | 50.2 +/- 1.6 | - | - | 39.6 +/- 6.8 | - | - | 33.4 +/- 1.8 | - | - |
| Vanilla LM | 65.2 +/- 6.8 | 71.7 +/- 6.8 | 77.6 +/- 4.1 | 56.7 +/- 4.4 | 59.3 +/- 3.8 | 62.5 +/- 3.3 | 36.3 +/- 2.7 | 40.5 +/- 2.9 | 43.6 +/- 3.1 |
| DiLM | 72.5 +/- 5.9 | 76.3 +/- 4.6 | 80.3 +/- 2.8 | 58.8 +/- 5.2 | 62.2 +/- 3.3 | 64.4 +/- 2.6 | 39.7 +/- 2.7 | 44.8 +/- 3.1 | 48.7 +/- 2.6 |
| Full dataset | 92.7 | 92.7 | 92.7 | 89.6 | 89.6 | 89.6 | 86.7 | 86.7 | 86.7 |

## Table 2: Cross-Model Generalization, DPC=20

`BERT_BASE (S)` is the source model used for DiLM gradient matching and K-centers feature extraction.

| Dataset | Model | Random | K-centers | DiLM |
|---|---|---:|---:|---:|
| SST-2 | BERT_BASE (S) | 70.3 +/- 6.8 | 79.8 +/- 3.5 | 80.3 +/- 2.8 |
| SST-2 | RoBERTa_BASE | 74.4 +/- 5.3 | 73.9 +/- 5.2 | 78.1 +/- 3.8 |
| SST-2 | BERT_LARGE | 74.7 +/- 8.4 | 80.4 +/- 9.1 | 83.1 +/- 6.2 |
| SST-2 | XLNet_BASE | 69.9 +/- 6.2 | 71.8 +/- 5.8 | 77.9 +/- 4.7 |
| QQP | BERT_BASE (S) | 59.1 +/- 3.8 | 62.6 +/- 2.7 | 64.4 +/- 2.6 |
| QQP | RoBERTa_BASE | 60.1 +/- 4.0 | 63.9 +/- 3.2 | 66.4 +/- 2.3 |
| QQP | BERT_LARGE | 58.8 +/- 6.9 | 59.0 +/- 8.9 | 62.9 +/- 8.6 |
| QQP | XLNet_BASE | 59.1 +/- 3.5 | 60.9 +/- 3.0 | 64.4 +/- 2.2 |
| MNLI-m | BERT_BASE (S) | 40.1 +/- 3.2 | 45.3 +/- 3.0 | 48.7 +/- 2.6 |
| MNLI-m | RoBERTa_BASE | 39.6 +/- 2.5 | 44.5 +/- 2.6 | 45.0 +/- 2.8 |
| MNLI-m | BERT_LARGE | 40.9 +/- 4.5 | 48.7 +/- 4.2 | 49.6 +/- 4.4 |
| MNLI-m | XLNet_BASE | 39.0 +/- 2.0 | 43.5 +/- 2.7 | 44.7 +/- 2.7 |

## Appendix Table 7: Cross-Model Generalization, DPC=5

| Dataset | Model | Random | K-centers | DiLM |
|---|---|---:|---:|---:|
| SST-2 | BERT_BASE (S) | 58.1 +/- 5.2 | 70.8 +/- 4.1 | 72.5 +/- 5.9 |
| SST-2 | RoBERTa_BASE | 60.6 +/- 7.6 | 74.2 +/- 4.9 | 75.1 +/- 4.6 |
| SST-2 | BERT_LARGE | 60.4 +/- 8.4 | 70.0 +/- 8.2 | 73.7 +/- 8.4 |
| SST-2 | XLNet_BASE | 57.0 +/- 5.5 | 66.4 +/- 5.0 | 69.5 +/- 6.6 |
| QQP | BERT_BASE (S) | 51.5 +/- 5.6 | 60.7 +/- 3.8 | 58.8 +/- 5.2 |
| QQP | RoBERTa_BASE | 52.5 +/- 6.0 | 63.9 +/- 3.3 | 62.4 +/- 3.7 |
| QQP | BERT_LARGE | 53.3 +/- 6.7 | 58.3 +/- 5.8 | 58.8 +/- 5.7 |
| QQP | XLNet_BASE | 52.6 +/- 5.2 | 62.6 +/- 3.1 | 60.2 +/- 4.6 |
| MNLI-m | BERT_BASE (S) | 35.6 +/- 2.1 | 36.2 +/- 2.4 | 39.7 +/- 2.7 |
| MNLI-m | RoBERTa_BASE | 35.8 +/- 2.1 | 37.4 +/- 2.1 | 38.8 +/- 3.0 |
| MNLI-m | BERT_LARGE | 36.9 +/- 2.8 | 37.4 +/- 2.9 | 41.5 +/- 3.7 |
| MNLI-m | XLNet_BASE | 35.4 +/- 1.4 | 37.0 +/- 1.5 | 37.3 +/- 1.9 |

## Appendix Table 8: Cross-Model Generalization, DPC=10

| Dataset | Model | Random | K-centers | DiLM |
|---|---|---:|---:|---:|
| SST-2 | BERT_BASE (S) | 64.3 +/- 7.4 | 75.9 +/- 4.7 | 76.3 +/- 4.6 |
| SST-2 | RoBERTa_BASE | 68.6 +/- 7.1 | 74.6 +/- 5.6 | 77.1 +/- 4.1 |
| SST-2 | BERT_LARGE | 67.2 +/- 8.5 | 76.6 +/- 8.4 | 79.2 +/- 7.8 |
| SST-2 | XLNet_BASE | 63.7 +/- 7.5 | 68.0 +/- 6.1 | 74.2 +/- 4.9 |
| QQP | BERT_BASE (S) | 56.0 +/- 4.8 | 60.9 +/- 3.1 | 62.2 +/- 3.3 |
| QQP | RoBERTa_BASE | 56.4 +/- 5.3 | 64.0 +/- 2.7 | 63.9 +/- 4.3 |
| QQP | BERT_LARGE | 53.7 +/- 8.5 | 59.4 +/- 5.6 | 60.6 +/- 7.5 |
| QQP | XLNet_BASE | 55.0 +/- 4.5 | 61.4 +/- 3.2 | 62.8 +/- 2.2 |
| MNLI-m | BERT_BASE (S) | 37.7 +/- 2.6 | 41.8 +/- 3.2 | 44.8 +/- 3.1 |
| MNLI-m | RoBERTa_BASE | 37.1 +/- 2.2 | 42.1 +/- 2.6 | 40.9 +/- 2.6 |
| MNLI-m | BERT_LARGE | 39.7 +/- 3.6 | 43.4 +/- 4.4 | 45.4 +/- 4.1 |
| MNLI-m | XLNet_BASE | 37.0 +/- 1.4 | 41.5 +/- 2.6 | 40.6 +/- 1.9 |

## Table 3: SST-2 5-Shot In-Context Learning, DPC=5

| Model | Random | K-centers | DiLM |
|---|---:|---:|---:|
| GPT-2-XL (1.5B) | 64.8 +/- 12.0 | 64.8 +/- 13.3 | 71.1 +/- 13.0 |
| OPT (2.7B) | 89.3 +/- 5.9 | 91.5 +/- 3.1 | 92.7 +/- 1.9 |
| Llama 2 (7B) | 93.6 +/- 2.9 | 94.6 +/- 0.7 | 95.1 +/- 0.7 |

## Table 4: Ablation Study, DiLM, DPC=5

RT = representative teacher.
DMS = diverse mini-batch sampling.
Selection = K-centers sample selection during synthetic dataset generation.

| RT | DMS | Selection | SST-2 | QQP | MNLI-m |
|---|---|---|---:|---:|---:|
| yes | yes | yes | 72.5 +/- 5.9 | 58.8 +/- 5.2 | 39.7 +/- 2.7 |
| no | yes | yes | 70.9 +/- 5.9 | 57.6 +/- 5.0 | 39.5 +/- 2.8 |
| yes | no | yes | 71.3 +/- 5.6 | 57.5 +/- 4.4 | 38.8 +/- 3.0 |
| yes | yes | no | 65.2 +/- 7.0 | 53.9 +/- 5.6 | 37.9 +/- 3.2 |

## Notes for Our Repository

Current notebooks target AG News. The DiLM paper does not report AG News metrics, so there are no paper-exact targets for:

- `01_full_data_baseline.ipynb`
- `02_random_coreset_baseline.ipynb`
- `03_stratified_random_baseline.ipynb`
- `04_kcenter_tfidf_baseline.ipynb`
- `05_kcenter_embedding_baseline.ipynb`

For AG News, expected metrics must be established by our own full-data and baseline runs. If the goal is exact paper reproduction, add separate notebooks or scripts for `SST-2`, `QQP`, and `MNLI-m` using the DiLM repository protocol.

