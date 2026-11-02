# Fake Job Posting Detection

Binary text classification to detect fraudulent job postings from the *Employment Scam* / fake job posting dataset.

## Pipeline

### 1. Data inspection
- Reports sample/feature counts, `.info()`, missing values, and class balance for the `fraudulent` label.

### 2. Cleaning & feature building
- Drops high-missing columns (`salary_range`, `department`, `in_balanced_dataset`).
- Fills text columns with empty strings.
- Concatenates `title`, `company_profile`, `description`, `requirements`, and `benefits` into `full_text`.

### 3. Preprocessing
- Lowercasing, punctuation removal, tokenization, English stopword removal, and WordNet lemmatization.
- TF-IDF vectorization (max 20 000 features).

### 4. Modeling
Stratified 80/20 train/test split (`random_state=42`), all models using `class_weight='balanced'`:

- **Logistic Regression** (`liblinear`, C = 1.0)
- **SVM — linear** kernel
- **SVM — RBF** kernel
- **KNN** — hyperparameters tuned with 5-fold `GridSearchCV` on F1 (`n_neighbors` ∈ {1,3,…,15}, `weights` ∈ {uniform, distance})

### 5. Evaluation
- Classification reports and timing for every model.
- A 4-panel confusion matrix figure with false-negative and false-positive counts.
- A metrics table computed from confusion matrices: accuracy, error rate, sensitivity/recall, specificity, precision, and F1.

`fraudulent` is mapped `t → 1`, `f → 0`.

## Requirements

```
pandas numpy nltk scikit-learn scipy matplotlib tabulate
```

The dataset (`DataSet.csv`) is expected in the working directory.

## Run

```bash
python fake-job-detection.py
```
