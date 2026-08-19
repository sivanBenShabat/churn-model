# Churn Model

Customer churn prediction for a telecom provider, built on the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset. A Logistic Regression model estimates each customer's probability of churning and recommends whether a retention outreach is worthwhile.

## Business logic

Retention teams have limited capacity and every outreach has a cost, so the model's decision threshold is chosen to minimize total business cost rather than to maximize a generic accuracy metric:

- **Missing an actual churner (false negative): 1,200 NIS** — lost customer revenue.
- **Retention offer sent to a customer who wouldn't have churned (false positive): 60 NIS** — cost of an unnecessary discount/offer.

Because a missed churner costs 20x more than a wasted offer, the optimal decision threshold is much lower than the default 0.5 — the model should flag customers even at low-to-moderate churn probability, trading precision for recall. See [Results](#results) below.

## Project structure

```
churn-model/
├── data/                  # raw, cleaned, and feature-engineered datasets (generated)
├── models/                # trained model artifact (generated)
├── plots/                 # EDA and evaluation charts (generated)
├── src/
│   ├── download_data.py   # download the raw dataset
│   ├── eda.py              # exploratory data analysis + plots
│   ├── clean.py             # data cleaning
│   ├── features.py          # feature engineering + one-hot encoding
│   ├── split_scale.py       # train/test split + StandardScaler
│   ├── train.py              # train the Logistic Regression model
│   ├── evaluate.py           # metrics + cost-based threshold optimization
│   └── serve.py               # FastAPI prediction endpoint
├── requirements.txt
└── README.md
```

## Pipeline

Run each step from the project root, in order, with the virtual environment activated.

### 0. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1. Download the data

```bash
python src/download_data.py
```
Saves the raw dataset to `data/telco_churn.csv` (7,043 rows).

### 2. Exploratory data analysis

```bash
python src/eda.py
```
Prints shape, dtypes, missing values, and churn rate. Saves `plots/churn_class_balance.png` and `plots/tenure_by_churn.png`.

### 3. Clean the data

```bash
python src/clean.py
```
Converts `TotalCharges` to numeric, drops the 11 rows where it's blank (all `tenure = 0`, no real value to impute), drops `customerID`, and encodes `Churn` as 0/1. Saves `data/telco_clean.csv`.

### 4. Feature engineering

```bash
python src/features.py
```
Adds `services_count` (number of active services) and `new_customer` (`tenure < 6`), then one-hot encodes categorical columns (`drop_first=True`). Saves `data/telco_features.csv`.

### 5. Train/test split + scaling

```bash
python src/split_scale.py
```
80/20 split, stratified on `Churn`, `random_state=42`. `StandardScaler` is **fit only on the training set** to avoid data leakage, then applied to both sets. Saves `data/X_train.csv`, `data/X_test.csv`, `data/y_train.csv`, `data/y_test.csv`, and `data/scaler.joblib`.

### 6. Train the model

```bash
python src/train.py
```
Trains a `LogisticRegression(class_weight="balanced", max_iter=1000)` on the training set and saves `models/churn_model.joblib`.

### 7. Evaluate

```bash
python src/evaluate.py
```
Computes Precision/Recall/F1/ROC-AUC at threshold 0.5, saves `plots/confusion_matrix.png`, and searches for the cost-minimizing threshold using the false negative / false positive costs above.

### 8. Serve predictions

```bash
python src/serve.py
```
Starts a FastAPI server on `http://localhost:8000`. `POST /predict` accepts a raw customer JSON record, runs the same preprocessing pipeline used in training, and returns:

```json
{"churn_risk": 0.822, "recommendation": "לפנות ללקוח"}
```

## Results

**Model:** Logistic Regression, `class_weight="balanced"` | **ROC-AUC: 0.838**

| Threshold | Precision | Recall | F1 | Total error cost |
|---|---|---|---|---|
| 0.50 (default) | 0.494 | 0.800 | 0.611 | 108,360 NIS |
| **0.12 (cost-optimal)** | 0.359 | 0.984 | — | **46,680 NIS** |

Using the cost-optimal threshold instead of 0.5 saves **61,680 NIS** on the test set by catching nearly all at-risk customers (98.4% recall), which is worthwhile given the 20:1 cost asymmetry between a missed churner and an unnecessary retention offer.

**Strongest churn-reducing factors:** longer tenure, two-year contracts, higher monthly charges.
**Strongest churn-increasing factors:** fiber optic internet, new customers (tenure < 6 months), streaming add-ons.
