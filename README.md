# Credit Default Risk Prediction with MLOps

An academic MLOps mini-project: two baseline models predict whether a credit card customer will default next month. Every training run is tracked in **MLflow**, the best model is registered in the **MLflow Model Registry**, served by a **FastAPI** prediction service, and the whole stack runs with **Docker Compose**.

> **Decision-support prototype, not an automated credit approval system.** See [Ethical Considerations](#ethical-considerations).

---

## Business Problem

> *"Can we identify customers who are at higher risk of defaulting on their credit card payment next month?"*

A financial institution can use a risk score as a **monitoring and decision-support tool**: flag higher-risk customers early so a human analyst can review them and consider preventive actions (a reminder, a payment plan, a conversation). The model supports people; it does not decide who receives credit.

---

## Dataset

**Default of Credit Card Clients Dataset** (Kaggle, `uciml/default-of-credit-card-clients-dataset`), originally from the UCI Machine Learning Repository. It describes credit card clients in Taiwan (2005): 30,000 customers and 25 columns.

| Group | Columns | Meaning |
|---|---|---|
| Identifier | `ID` | Row number - **removed, never used as a feature** |
| Credit | `LIMIT_BAL` | Credit limit (NT dollars) |
| Demographics | `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Personal attributes (sensitive, see ethics) |
| Repayment status | `PAY_0`, `PAY_2` ... `PAY_6` | -2/-1/0 = on time or no use, 1-9 = months late (Sep → Apr) |
| Bills | `BILL_AMT1` ... `BILL_AMT6` | Monthly bill statements |
| Payments | `PAY_AMT1` ... `PAY_AMT6` | Monthly amounts paid |
| **Target** | `default.payment.next.month` | 1 = default next month, 0 = no default |

The code automatically detects the target under any of these names: `default.payment.next.month`, `default payment next month`, `default_payment_next_month`. It also renames `PAY_1` to `PAY_0` if your version uses that name.

### Where to put the CSV

1. Download the dataset from Kaggle: <https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset> (the file is called `UCI_Credit_Card.csv`).
2. Save it as:

```
data/raw/default_credit_card_clients.csv
```

If you forget to rename it, it still works as long as it is the **only** CSV in `data/raw/`.

### Optional - download automatically with the Kaggle API

1. Create a free Kaggle account, go to <https://www.kaggle.com/settings/api> and click **Generate New Token**.
2. Create your `.env` file and paste the token:

```bash
cp .env.example .env
nano .env        # fill in KAGGLE_API_TOKEN=...   (save: Ctrl+O, Enter, Ctrl+X)
```

3. Download (inside Docker, no local Python needed):

```bash
docker compose --profile train run --rm trainer python -m src.download_data
ls data/raw
```

You can also skip step 3: `docker compose --profile train run --rm trainer` downloads the CSV automatically when it is missing and a token is present. The code (`src/download_data.py`) also accepts the older `KAGGLE_USERNAME` + `KAGGLE_KEY` values from `kaggle.json`.

`.env` holds a secret: it is in `.gitignore` and must never be committed. Only `.env.example` (empty) goes to Git.

The dataset is **not** committed to Git (see `.gitignore`).

---

## Architecture

```
 Kaggle Dataset  (data/raw/default_credit_card_clients.csv)
       |
       v
 Data Preparation        src/data.py, src/features.py
       |
       v
 Training                src/train.py
 (Logistic Regression / Random Forest)
       |
       v
 MLflow Tracking         one Run per model: parameters + metrics + artifacts + model
       |
       v
 Model Comparison        highest ROC-AUC wins
       |
       v
 MLflow Model Registry   credit_default_model  (alias: @champion)
       |
       v
 FastAPI                 api/main.py  ->  POST /predict
       |
       v
 Docker Compose          mlflow  |  trainer  |  api
```

```
            Your machine (Docker Desktop)
 ┌───────────────────────────────────────────────────────────┐
 │  Docker Compose network                                   │
 │                                                           │
 │  ┌──────────┐   logs runs,     ┌──────────────────────┐   │
 │  │ trainer  │ ───────────────> │ mlflow   :5000       │   │
 │  │ (runs    │   registers      │ SQLite + artifacts   │ <──── http://localhost:5000
 │  │  once)   │   model          │ (volume mlflow_data) │   │
 │  └──────────┘                  └──────────────────────┘   │
 │       ^ reads CSV                      ^ loads @champion  │
 │   ./data/raw                   ┌──────────────────────┐   │
 │                                │ api      :8000       │ <──── http://localhost:8000/docs
 │                                │ FastAPI              │   │
 │                                └──────────────────────┘   │
 └───────────────────────────────────────────────────────────┘
```

Containers talk to MLflow at `http://mlflow:5000`; your browser uses `http://localhost:5000`.

---

## Project Structure

```
credit-default-mlops/
├── README.md
├── MODEL_CARD.md             # governance: purpose, limits, performance, fairness
├── requirements.txt          # pinned Python dependencies
├── pytest.ini                # test settings
├── .gitignore
├── .env.example              # template for Kaggle credentials (copy to .env)
├── .dockerignore
├── Dockerfile                # one image shared by all services
├── docker-compose.yml        # mlflow + trainer + api
├── .github/workflows/ci.yml  # bonus: runs tests on every push
├── data/raw/.gitkeep         # put the Kaggle CSV here
├── notebooks/01_eda.ipynb    # exploratory data analysis
├── src/
│   ├── config.py             # ALL settings in one place
│   ├── data.py               # load CSV, detect target, drop ID, split
│   ├── download_data.py      # optional: get the CSV via the Kaggle API
│   ├── features.py           # derived features + preprocessing pipeline
│   ├── evaluate.py           # metrics and plots
│   ├── fairness.py           # fairness check per group (sex, age, marriage, education)
│   ├── train.py              # train, log to MLflow, pick the best
│   └── register_model.py     # Model Registry + @champion alias
├── api/
│   ├── main.py               # FastAPI app
│   └── schemas.py            # request / response validation
├── tests/
│   ├── conftest.py           # synthetic test data
│   ├── test_data.py
│   ├── test_api.py
│   └── test_fairness.py
└── artifacts/.gitkeep        # plots and summaries appear here after training
```

---

## Machine Learning Approach

| Step | Choice | Why |
|---|---|---|
| Split | 80% train / 20% test, **stratified**, `random_state=42` | Keeps the same default rate in both sets; reproducible |
| Categorical | `SEX`, `EDUCATION`, `MARRIAGE` → one-hot encoding | They are codes, not quantities |
| Cleaning | `EDUCATION` 0/5/6 → 4 (others), `MARRIAGE` 0 → 3 (others) | Undocumented codes |
| Numeric | Standard scaling | Needed by Logistic Regression |
| Derived features | `AVG_BILL_AMT`, `AVG_PAY_AMT`, `NUM_DELAYED_PAYMENTS` | Easy to explain to business users |
| Models | Logistic Regression, Random Forest | Two simple, well-known baselines |
| Imbalance | `class_weight="balanced"` | Defaulters are the minority class |
| Selection | **ROC-AUC** | Measures ranking quality independently of the threshold |

All preprocessing is **inside one scikit-learn Pipeline**, so the API applies exactly the same transformations as training.

### Metrics

We log **ROC-AUC, recall, precision, F1 and accuracy**. Accuracy alone is misleading: most customers do not default, so a model that always says "no default" looks accurate but catches nobody.

**Why recall matters here:** a **false negative** means we predicted "low risk" for a customer who actually defaults, so the institution misses the chance to act early. Recall measures how many real defaulters we catch. But higher recall usually means more **false positives** (customers wrongly flagged), which has a cost in analyst time and customer experience. No single metric decides the business action; the threshold (0.5 by default in `src/config.py`) should be agreed with the business.

Each run also saves a **confusion matrix**, a **ROC curve** and a **precision-recall curve**.

---

## MLflow Experiment Tracking

Experiment name: **`credit-default-risk`**. Every `python -m src.train` creates **one MLflow Run per model**:

| Logged | Content |
|---|---|
| Parameters | model type, hyperparameters, random state, test size, stratification, class weighting, threshold |
| Metrics | accuracy, precision, recall, f1, roc_auc, `fairness_*_ratio` (one per attribute and metric) |
| Artifacts | `plots/confusion_matrix.png`, `plots/roc_curve.png`, `plots/precision_recall_curve.png`, `summary/run_summary.json`, `fairness/fairness_report.json`, `fairness/fairness_by_group.png` |
| Model | the full scikit-learn pipeline (MLflow sklearn flavor) |
| Tags | `selected_as_champion = true/false`, `fairness_needs_review` |

MLflow uses a **SQLite** database for metadata and a folder for artifacts, both stored in the Docker volume `mlflow_data`, so nothing is lost when containers restart.

A copy of the plots is also written to `./artifacts/` on your machine, plus `artifacts/model_comparison.json` and `artifacts/roc_comparison.png` (both models on one chart - useful for the slides).

---

## Model Registry

After both runs finish, the model with the highest ROC-AUC is registered as **`credit_default_model`** and receives the alias **`@champion`**. Each retraining creates a new version and moves the alias. (Aliases replace the deprecated "Staging/Production" stages.)

The API loads `models:/credit_default_model@champion`. If the alias is missing, it falls back to the **latest registered version**.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Project status |
| GET | `/health` | `{"status": "healthy", "model_loaded": true}` |
| POST | `/predict` | Default probability for one customer |
| POST | `/reload` | Reload the `@champion` model after retraining |

Swagger documentation: **<http://localhost:8000/docs>** (the example request is pre-filled - click *Try it out* → *Execute*).

Inputs are validated with Pydantic (e.g. `AGE` must be 18-100, `SEX` must be 1 or 2, `PAY_*` must be -2 to 9). Invalid input returns HTTP 422. If no model is registered yet, `/predict` returns HTTP 503.

---

## Docker

One `Dockerfile` builds one image used by three services:

| Service | What it does | Port |
|---|---|---|
| `mlflow` | MLflow tracking server + Model Registry | `localhost:5000` |
| `trainer` | Runs `python -m src.train` once, then stops | - |
| `api` | FastAPI + uvicorn | `localhost:8000` |

The trainer has the Compose profile `train`, so `docker compose up` does not start it by accident; you run it on demand. You only need **Git, Docker Desktop and the CSV** - no local Python.

---

## How to Run

### 0. Prerequisites

- Git
- Docker Desktop (running)
- The Kaggle CSV

### 1. Get the code

```bash
git clone https://github.com/hugomacedolima18-stack/credit-default-mlops.git
cd credit-default-mlops
```

### 2. Add the data

```bash
cp ~/Downloads/UCI_Credit_Card.csv data/raw/default_credit_card_clients.csv
ls data/raw
```

### 3. Build the image (first time takes a few minutes)

```bash
docker compose build
```

### 4. Start MLflow

```bash
docker compose up -d mlflow
docker compose ps
```

Wait until `credit-mlflow` shows `healthy`, then open **<http://localhost:5000>**.

### 5. Train the models

```bash
docker compose --profile train run --rm trainer
```

You will see a comparison table at the end and `Registered 'credit_default_model' version 1`.

### 6. Start the API

```bash
docker compose up -d api
curl http://localhost:8000/health
```

Open **<http://localhost:8000/docs>**.

### 7. Stop everything

```bash
docker compose down          # stops containers, KEEPS the MLflow data
docker compose down -v       # stops containers AND DELETES the MLflow data
```

### Retraining

```bash
docker compose --profile train run --rm trainer          # creates new runs + a new model version
curl -X POST http://localhost:8000/reload
```

### Running without Docker (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d mlflow              # MLflow still runs in Docker
python -m src.train                      # uses http://localhost:5000 by default
uvicorn api.main:app --port 8000
```

For the notebook: `pip install notebook` then `jupyter notebook notebooks/01_eda.ipynb`.

---

## Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 46000, "BILL_AMT2": 45000, "BILL_AMT3": 43000,
    "BILL_AMT4": 30000, "BILL_AMT5": 29000, "BILL_AMT6": 28000,
    "PAY_AMT1": 0, "PAY_AMT2": 1500, "PAY_AMT3": 1200,
    "PAY_AMT4": 1100, "PAY_AMT5": 1000, "PAY_AMT6": 1000
  }'
```

Response format (the numbers depend on your trained model):

```json
{
  "prediction": 1,
  "label": "high_default_risk",
  "default_probability": 0.73,
  "threshold": 0.5,
  "model_version": "1",
  "model_uri": "models:/credit_default_model@champion",
  "disclaimer": "Academic decision-support prototype. ..."
}
```

---

## Testing

The tests use a small **synthetic** dataset, so they need neither the Kaggle file nor a running MLflow.

With Docker:

```bash
docker compose --profile train run --rm trainer pytest
```

Locally:

```bash
pytest
```

They check that the dataset loads, the target is detected under its different names, `ID` is removed, the split is stratified, the derived features are correct, `/health` returns 200, `/predict` returns the expected fields, invalid input returns 422 and a missing model returns 503.

**Bonus - CI:** `.github/workflows/ci.yml` runs the tests and builds the Docker image on every push to GitHub.

---

## Fairness Check & Model Card

Credit scoring is **high-risk under the EU AI Act**, so every training run also checks whether the model works equally well for different groups (`src/fairness.py`):

- Groups: **SEX**, **AGE** band (<30, 30-39, 40-49, 50+), **MARRIAGE**, **EDUCATION**.
- Per group: **recall** (defaulters caught), **false positive rate** (good customers wrongly flagged), **flag rate**.
- Comparison: *ratio = lowest / highest*. **80% rule:** a ratio below 0.8 is listed in `fairness_needs_review` for a human to check. Groups with fewer than 30 cases for a metric are not compared.
- Everything is logged to MLflow (artifacts in `fairness/`, metrics `fairness_*_ratio`).

Result for the champion: **sex and age pass**; **education and marital status need review** (e.g. high-school customers get more false alarms than graduate-school customers). Details and next steps are in **[MODEL_CARD.md](MODEL_CARD.md)**.

---

## Ethical Considerations

- **Responsible use:** the output is a risk *score* for human review. It must not be used to automatically approve, reject or reprice credit.
- **Fairness and bias:** the data contains `SEX`, `AGE`, `MARRIAGE` and `EDUCATION`. A model can learn historical patterns that disadvantage certain groups. Before any real use, compare recall and false-positive rates across groups, and test whether removing these variables changes the results. Note that other variables can act as proxies for them.
- **Explainability:** people affected by credit-related decisions should be able to understand them. Logistic Regression coefficients are easy to explain; Random Forest needs tools such as feature importance or SHAP.
- **Privacy:** financial and demographic data is personal data. A real system needs a legal basis, data minimisation, access control and retention rules (GDPR). This project uses a public, anonymised dataset and never commits data to Git.
- **Regulation and compliance:** credit scoring is a heavily regulated area. In the EU, AI systems used to evaluate the creditworthiness of natural persons are classified as high-risk under the AI Act, which requires risk management, documentation, human oversight and monitoring.
- **Human oversight and monitoring:** a person reviews every flagged case; after deployment, performance and data drift must be monitored.

---

## Limitations

- Data from Taiwan in 2005: it does not represent today's customers or other countries.
- Only two baseline models, with no hyperparameter tuning.
- Fixed threshold of 0.5, not optimised for business costs.
- No probability calibration.
- One train/test split (no cross-validation).
- MLflow uses SQLite and has no authentication: fine for a laptop, not for production.
- No monitoring of predictions or data drift after deployment.

---

## Future Improvements

- Cross-validation and hyperparameter tuning (tracked in MLflow).
- Choose the threshold with the business, based on the cost of false negatives versus false positives.
- Retrain without sensitive attributes and compare the fairness gaps; explainability (SHAP) as an MLflow artifact.
- Probability calibration.
- Data validation of incoming requests and drift monitoring (e.g. Evidently).
- A simple Streamlit front-end for analysts.
- PostgreSQL + object storage for MLflow, and authentication for the API.
