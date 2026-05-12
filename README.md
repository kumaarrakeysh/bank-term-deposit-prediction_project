# Bank Term Deposit Subscription Prediction

## Project Overview
This project predicts whether a bank customer will subscribe to a term deposit.

## Models Used
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking Classifier

## Run Project

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Environment

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Training
```bash
python main.py
```

### 5. Run Streamlit App
```bash
streamlit run app/streamlit_app.py
```