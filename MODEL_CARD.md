# Model Card - `credit_default_model`

> A model card is the "information leaflet" of a model: what it does, how well, for whom, and where it must **not** be used. (Governance deliverable - see class 4, *Practical Governance Deliverables*.)

| | |
|---|---|
| **Model name** | `credit_default_model` (MLflow Model Registry, alias `@champion`) |
| **Current champion** | Random Forest (ROC-AUC 0.775). Promotion rule: a new version becomes `@champion` only if its ROC-AUC beats the current champion (`src/register_model.py`) |
| **Owners** | Fabrício Olo, Hugo Lima, Maria Teresa Neves, Olívia Rua, Pedro Martim Lota |
| **Status** | Academic prototype (Porto Business School, MLOps mini-project, Oct 2026) |
| **Code** | `src/train.py` (training), `src/fairness.py` (fairness check), `api/main.py` (serving) |

---

## 1. Intended use

- **Purpose:** estimate the probability that a credit card customer **defaults on next month's payment**.
- **Users:** credit risk analysts.
- **How it is used:** flagged customers are **reviewed by a human analyst**, who can decide on a preventive action (reminder, payment plan, conversation).
- **Human in the loop:** every API response carries a disclaimer that the score must be reviewed by a person.

## 2. Out of scope (must NOT be used for)

- Automatically **approving, rejecting or repricing** credit.
- Any decision about a person **without human review**.
- Customers outside the population it was trained on (see *Data*).

## 3. Data

| | |
|---|---|
| **Source** | *Default of Credit Card Clients* (UCI / Kaggle), public and anonymised |
| **Population** | 30,000 credit card customers, **Taiwan, 2005** |
| **Features** | credit limit, sex, education, marital status, age, 6 months of repayment status, bills and payments + 3 derived features |
| **Target** | default next month (22.1% = yes) |
| **Split** | 80% train / 20% test, stratified, `random_state=42` |
| **Personal data** | yes (financial + demographic). The CSV is **never committed to Git** (`.gitignore`). |

## 4. Performance (test set, 6,000 customers, threshold 0.5)

| Metric | Logistic Regression | **Random Forest (champion)** |
|---|---|---|
| ROC-AUC | 0.745 | **0.775** |
| Recall | 0.561 | **0.580** |
| Precision | 0.470 | **0.504** |
| F1 | 0.512 | **0.539** |
| Accuracy | 0.763 | **0.781** |

In business terms: the champion catches **770 of 1,327 defaulters** (58%); about **half of the flags are real defaulters** (759 false alarms).

## 5. Fairness check

Computed automatically on every training run (`src/fairness.py`) and logged to MLflow (`fairness/fairness_report.json`, `fairness/fairness_by_group.png`, metrics `fairness_*_ratio`, tag `fairness_needs_review`).

**Method:** for each group we compute **recall** (defaulters caught), **false positive rate** (good customers wrongly flagged) and **flag rate**. We compare groups with *ratio = lowest / highest*. **Rule of thumb (80% rule): ratio < 0.8 -> needs human review.** Groups with fewer than 30 cases for a metric are reported but not compared (too small to be reliable).

**Results for the champion (Random Forest):**

| Attribute | Recall ratio | False-positive ratio | Flag-rate ratio | Verdict |
|---|---|---|---|---|
| SEX | 1.00 | 0.83 | 0.88 | ✅ passes |
| AGE (<30, 30-39, 40-49, 50+) | 0.92 | 0.82 | 0.83 | ✅ passes |
| MARRIAGE | 1.00 | **0.72** | 0.93 | ⚠️ review |
| EDUCATION | 0.90 | **0.32** | **0.24** | ⚠️ review |

**Interpretation:**

- **Sex and age:** the model catches defaulters equally well (recall 0.58 for women and men) and false-alarm rates are close. No action needed beyond monitoring.
- **Marital status:** the gap comes from the small "others" group (75 customers): 22% false alarms vs 16% for married/single.
- **Education:** customers with **high-school education get more false alarms (20%) than graduate-school customers (12%)**. Part of this reflects a real difference in default rates in the data (26% vs 19%), but it also means that, at the same threshold, good customers with less education are flagged more often. The "others" group (82 customers, only ~5 defaulters) is too small to judge.

**Decision / next steps before any real use:**

1. A human (risk + compliance) reviews the EDUCATION and MARRIAGE gaps.
2. Experiment: retrain **without the sensitive attributes** and compare performance and gaps (note: other variables can act as proxies).
3. Consider group-aware thresholds only with legal review.
4. Keep monitoring these ratios on every retraining.

## 6. Risks and limitations

- Data is from **Taiwan in 2005**: it does not represent today's customers or other countries.
- One train/test split, no hyperparameter tuning, fixed threshold 0.5 (not yet agreed with the business).
- Probabilities are not calibrated.
- Random Forest is harder to explain than Logistic Regression (SHAP is a future step).
- No monitoring of production data or drift yet.
- MLflow runs locally (SQLite, no authentication): fine for a prototype, not for production.

## 7. Regulation and governance

- **EU AI Act:** AI systems that evaluate the creditworthiness of natural persons are **high-risk**. This requires risk management, documentation (this card), human oversight, and monitoring.
- **GDPR:** financial and demographic data is personal data: legal basis, data minimisation, access control and retention rules are needed for real use.
- **Audit trail:** every model version in the MLflow Registry links to the run that produced it (parameters, metrics, fairness report, model file).

## 8. How to reproduce

```bash
docker compose build
docker compose up -d mlflow
docker compose --profile train run --rm trainer
```

Same data + same code + `random_state=42` = the same model and the same numbers.
