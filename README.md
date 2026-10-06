# Hidden Structural Leakage in Arabic Legal Document Classification

Code and public metadata for the paper *"Hidden Structural Leakage in Arabic Legal Document Classification: An Audit of Moroccan Court of Cassation Decisions"* (A. Es-Salmi, E. Smili, C. Loqman).

Keyword masking of chamber names, the usual remedy against label leakage, leaves the main leakage channels of Moroccan Court of Cassation decisions intact (bench composition: 98.5% accuracy from the last 400 characters; case-file number: 94.8% from a single digit). This repository contains the leakage audit, the structural masking procedure, and the experiments of the paper.

## Contents

| Path | Description |
|---|---|
| `mask_v1.py` | Keyword masking (v1): explicit chamber names and case-file type labels → `[MASKED]` |
| `mask_v2.py` | Structural masking (v2) and content-only version (v3) |
| `probes.py` | Leakage probes (header line, last 400 characters, file-number digit, year) |
| `run_xgboost.py` | TF-IDF + XGBoost, 5 seeds, random or chronological split |
| `run_encoder.py` | Fine-tuning of AraBERTv02 / CAMeLBERT-Mix, 5 seeds |
| `aggregate.py`, `mcnemar.py` | Mean ± s.d., paired t-tests, McNemar's tests |
| `shap_v2.py`, `shap_glosses.csv` | TreeSHAP analysis with data-driven leakage detection |
| `attention_fig10.py`, `leakage_keywords.txt` | Word-level [CLS] attention analysis |
| `common.py` | Data loading, normalisation, splits, metrics |
| `make_release_files.py` | Builds `release/` from a local copy of the corpus |
| `release/document_ids.csv` | Chamber, decision number, decision date and case-file number of the 11,166 decisions |
| `release/split_random.csv` | Train/validation/test assignment for seeds 42, 13, 7, 2024, 123 |
| `release/split_chronological.csv` | Per-chamber chronological split |
| `predictions/` | Test-set predictions of all models (`id`, `y_true`, `y_pred`) |

## Data

The decisions are publicly available on the judicial portal of the Moroccan Court of Cassation (<https://juriscassation.cspj.ma>). **The texts are not redistributed here** because they contain personal data (names of parties, judges, lawyers) that were not anonymised in this study. The processed corpora are available from the corresponding author upon reasonable request for research purposes, in accordance with Moroccan Law No. 09-08.

To rebuild the corpus, download the decisions listed in `release/document_ids.csv` and save each one as plain text in `CORPUS_PROCESSED/<chamber>/<file>`, using the `id` column as the relative path.

## Reproducing the experiments

```bash
pip install -r requirements.txt

# 1. Masking
python mask_v1.py                 # CORPUS_PROCESSED -> DATASET_MASKED (v1)
python mask_v2.py                 # DATASET_MASKED -> DATASET_MASKED_V2 (v2) and DATASET_BODY_V3 (v3)

# 2. Leakage probes
python probes.py --masked DATASET_MASKED
python probes.py --masked DATASET_MASKED_V2

# 3. Classification (5 seeds)
python run_xgboost.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_v2
python run_encoder.py --model aubmindlab/bert-base-arabertv02 --name arabertv2 \
       --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_v2 --keep-seed 42
python run_encoder.py --model CAMeL-Lab/bert-base-arabic-camelbert-mix --name camelbert \
       --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_v2

# 4. Chronological evaluation (XGBoost, 3 seeds)
python run_xgboost.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_chrono \
       --split release/split_chronological.csv --seeds 42 13 7

# 5. Statistics
python aggregate.py --out results_v2
python mcnemar.py --out results_v2 --seed 42

# 6. Explainability
python shap_v2.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 --out results_v2/shap
python attention_fig10.py --orig CORPUS_PROCESSED --masked DATASET_MASKED_V2 \
       --models results_v2/models --name arabertv2 --keywords leakage_keywords.txt --out results_v2
```

XGBoost builds trees with multiple threads, so results may differ by about ±0.1 percentage points across machines. The transformers were fine-tuned on a single NVIDIA GeForce RTX 2070 (8 GB).

## Citation

```bibtex
@article{essalmi2026leakage,
  author  = {Es-Salmi, Aicha and Smili, Elmehdi and Loqman, Chakir},
  title   = {Hidden Structural Leakage in Arabic Legal Document Classification: An Audit of Moroccan Court of Cassation Decisions},
  journal = {[Journal]},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Metadata files in `release/` and `predictions/`: CC BY 4.0.
