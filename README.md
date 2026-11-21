# Kazakh Profanity Detection in Product Reviews

**Deep Learning - Bonus Project**
**Author:** Nurbolat Nuray · **Student ID:** 25MD0171

A text classifier that reads one Kazakh review or comment and predicts whether it contains Kazakh profanity.

- **Input:** one comment or sentence
- **Output:** label `0` (no profanity) or `1` (profanity present), plus the model's probability and the decision threshold

The complete solution is a single notebook, **[`kazakh_profanity_detection.ipynb`](kazakh_profanity_detection.ipynb)**. It covers data cleaning and splitting, preprocessing, training, test evaluation, error analysis and inference, and it is committed with its outputs so it can be read without running it.

## Results at a glance

| Item | Value |
|---|---|
| Data | 3,061 labeled Kazakh app/service reviews and comments, 3,027 after cleaning |
| Synthetic data | none in train/val/test (a separate 19-sentence stress set is used for qualitative checks only) |
| Split | train 2,423 · val 302 · test 302, random seed 42, near-duplicates kept in the same split |
| Model | TF-IDF (character 2–4-grams + word 1–2-grams) → logistic regression, `C = 30` |
| Decision threshold | 0.42, selected on the validation set |
| **Test set (n = 302, balanced)** | **precision 0.952 · recall 0.921 · F1 0.936** (profanity class) |
| Confusion matrix | TN = 144 · FP = 7 · FN = 12 · TP = 139 |

The test set was used exactly once, after all settings and the threshold had been chosen on validation data.

## Dataset and labeling

- **Source:** `data/raw/kazakh_profanity_share.xlsx` (columns `kazsandra_id`, `text`, `label`, `split`). The ids follow the KazSAndRA naming scheme; the texts are user reviews and comments about apps and services, so they match the product-review setting of the task. The file was collected and labeled by a colleague (Amir) and shared with me.
- **Labels were checked against this task**, not taken as a general toxicity or sentiment label. Two clear labeling errors were corrected (listed in the notebook).
- **Labeling rules** (full table in notebook §2):
  - an obscene or insulting word makes the comment `1`;
  - masked (`қо***бас`), misspelled or Latin-script versions of such a word are also `1`;
  - for mixed Kazakh/Russian/English comments, the label depends only on whether a profane word is present;
  - a negative or angry review without a profane word is `0`;
  - the ambiguous word «мал» is `0` when it means livestock and `1` when used as an insult.
- **Cleaning and split** (notebook §4): emails, links, phone numbers and handles removed; reviewer ids are not written to the processed data; exact duplicates removed after normalisation; near-duplicates (character n-gram cosine > 0.8) grouped and kept in one split via `StratifiedGroupKFold`. The vocabulary is fitted on training data only.

Labeled examples, class distribution and the split table are printed in the notebook.

## Model

Text is normalised by a hand-written, stateless function (lowercasing, Latin→Cyrillic transliteration, Kazakh→Russian letter folding, restoring masked words, collapsing repeated letters). It is then converted into character and word TF-IDF features and classified by logistic regression. Unknown words are still covered by character n-grams, and long comments are truncated to the first `max_words` words. A small weighting scheme reduces the influence of the over-represented word «мал». `C`, n-gram length, weighting strength, `max_words` and the threshold were all chosen on the validation set (grid in `results/val_grid.csv`).

## Repository structure

```
kazakh_profanity_detection.ipynb    notebook with all code, explanations and outputs
requirements.txt                    pinned dependencies
data/raw/                           input dataset
data/processed/                     cleaned data with train/val/test assignment (no personal information)
data/split_info.json                split sizes, cleaning counts, seed
models/model.joblib                 trained pipeline: preprocessing, vocabulary, classifier
models/model_meta.json              selected parameters and threshold
results/                            test_metrics.json, confusion_matrix.png, test_predictions.csv,
                                    test_errors.csv, val_grid.csv, stress_predictions.csv
```

## How to run

**Training and evaluation (reproduces all results, about 30 s on CPU):**

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # Python 3.11+, tested on 3.12
jupyter notebook kazakh_profanity_detection.ipynb    # Kernel → Restart & Run All
```

In Google Colab: upload the repository (or clone it), run `!pip install -r requirements.txt`, then Run all.

**Prediction without retraining:** the trained model is in `models/`. Run notebook §1 (setup) and §3 (the `normalize` function, which the saved pipeline needs), then §9:

```python
predict("Бұл ойын зайбал екен")
# {'label': 1, 'probability': 0.9998, 'threshold': 0.42}
```

Section 9 also has a cell where you can type any custom comment.

## Error analysis and limitations

The notebook (§7) lists all 19 test errors and discusses five of them: heavily masked words, probable label noise, borderline slang, a long comment with a single profane word, and the ambiguity of «мал», each with a suggested improvement. Main limitations: a small and partly noisy dataset, a small test set (95 % interval about ±3 points), hand-made word lists, limited handling of Latin-script and heavily obfuscated text, and no context beyond one comment.

