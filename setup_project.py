"""
setup_project.py
================
Run this file inside your project folder:
    python setup_project.py

It will create the complete professional ML project structure.
"""

import os
import sys

# ─────────────────────────────────────────────
#  Project name = current folder name
# ─────────────────────────────────────────────
PROJECT_NAME = os.path.basename(os.getcwd())

# ─────────────────────────────────────────────
#  Folders to create
# ─────────────────────────────────────────────
FOLDERS = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
]

# ─────────────────────────────────────────────
#  File templates
# ─────────────────────────────────────────────

GITIGNORE = """# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
venv/
env/
.env/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data files (large)
data/raw/*.csv
data/raw/*.zip
data/raw/*.json
data/processed/*.csv

# Models (large)
models/*.pkl
models/*.h5
models/*.joblib

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
"""

REQUIREMENTS = """# Core ML
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn

# App
streamlit

# Utils
joblib
"""

README = f"""# {PROJECT_NAME.replace("-", " ").title()}

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

## Problem Statement
<!-- Describe the problem you are solving -->

## Dataset
- **Source:** Kaggle — [Dataset Name](link-here)
- **Rows:** ...
- **Features:** ...
- **Target:** ...

## Model Used
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| XGBoost | xx% | xx% | xx% | xx% |

## Project Structure
```
{PROJECT_NAME}/
├── data/
│   ├── raw/             # Original Kaggle dataset
│   └── processed/       # Cleaned & encoded data
├── notebooks/
│   ├── 01_EDA.ipynb     # Exploratory Data Analysis
│   └── 02_model.ipynb   # Model training & evaluation
├── src/
│   ├── preprocess.py    # Data cleaning functions
│   └── train.py         # Model training script
├── models/              # Saved model files (.pkl)
├── app.py               # Streamlit web app
├── requirements.txt
└── README.md
```

## How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/{PROJECT_NAME}.git
cd {PROJECT_NAME}

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Live Demo
- Streamlit Cloud: [link-here]
- HuggingFace Spaces: [link-here]

## Tech Stack
- Python 3.9+
- Scikit-learn
- XGBoost
- Pandas / NumPy
- Streamlit
- Matplotlib / Seaborn

## Results
<!-- Add confusion matrix screenshot or accuracy chart here -->

## Author
[Your Name] — [GitHub](https://github.com/YOUR_USERNAME) | [Upwork Profile](link)
"""

EDA_NOTEBOOK = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Exploratory Data Analysis\\n", "Project: """ + PROJECT_NAME + """"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "sns.set_style('whitegrid')\\n",
    "plt.rcParams['figure.figsize'] = (10, 6)\\n",
    "\\n",
    "df = pd.read_csv('../data/raw/YOUR_FILE.csv')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Shape and info\\n",
    "print('Shape:', df.shape)\\n",
    "print('\\\\nData Types:\\\\n', df.dtypes)\\n",
    "print('\\\\nMissing Values:\\\\n', df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistical summary\\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Target distribution\\n",
    "target_col = 'target'  # CHANGE THIS\\n",
    "df[target_col].value_counts().plot(kind='bar', color=['steelblue','salmon'])\\n",
    "plt.title('Target Distribution')\\n",
    "plt.xlabel('Class')\\n",
    "plt.ylabel('Count')\\n",
    "plt.tight_layout()\\n",
    "plt.savefig('../data/processed/target_dist.png')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation heatmap\\n",
    "plt.figure(figsize=(12, 8))\\n",
    "sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)\\n",
    "plt.title('Feature Correlation Matrix')\\n",
    "plt.tight_layout()\\n",
    "plt.savefig('../data/processed/correlation.png')\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""

MODEL_NOTEBOOK = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Model Training & Evaluation\\n", "Project: """ + PROJECT_NAME + """"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "import joblib\\n",
    "\\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\\n",
    "from sklearn.metrics import (accuracy_score, classification_report,\\n",
    "                              confusion_matrix, roc_auc_score)\\n",
    "from xgboost import XGBClassifier\\n",
    "\\n",
    "df = pd.read_csv('../data/raw/YOUR_FILE.csv')\\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Preprocessing ──────────────────────────────────\\n",
    "target_col = 'target'  # CHANGE THIS\\n",
    "\\n",
    "X = df.drop(columns=[target_col])\\n",
    "y = df[target_col]\\n",
    "\\n",
    "# Handle missing values\\n",
    "X = X.fillna(X.median(numeric_only=True))\\n",
    "\\n",
    "# Encode categoricals\\n",
    "for col in X.select_dtypes(include='object').columns:\\n",
    "    X[col] = LabelEncoder().fit_transform(X[col].astype(str))\\n",
    "\\n",
    "# Train-test split\\n",
    "X_train, X_test, y_train, y_test = train_test_split(\\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\\n",
    ")\\n",
    "print(f'Train: {X_train.shape}, Test: {X_test.shape}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Model Training ─────────────────────────────────\\n",
    "model = XGBClassifier(\\n",
    "    n_estimators=200,\\n",
    "    max_depth=4,\\n",
    "    learning_rate=0.05,\\n",
    "    subsample=0.8,\\n",
    "    colsample_bytree=0.8,\\n",
    "    use_label_encoder=False,\\n",
    "    eval_metric='logloss',\\n",
    "    random_state=42\\n",
    ")\\n",
    "\\n",
    "model.fit(X_train, y_train,\\n",
    "          eval_set=[(X_test, y_test)],\\n",
    "          verbose=50)\\n",
    "print('Training complete!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Evaluation ─────────────────────────────────────\\n",
    "y_pred = model.predict(X_test)\\n",
    "y_prob = model.predict_proba(X_test)[:, 1]\\n",
    "\\n",
    "print(f'Accuracy  : {accuracy_score(y_test, y_pred):.4f}')\\n",
    "print(f'ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}')\\n",
    "print()\\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix\\n",
    "cm = confusion_matrix(y_test, y_pred)\\n",
    "plt.figure(figsize=(6, 5))\\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\\n",
    "plt.title('Confusion Matrix')\\n",
    "plt.ylabel('Actual')\\n",
    "plt.xlabel('Predicted')\\n",
    "plt.tight_layout()\\n",
    "plt.savefig('../data/processed/confusion_matrix.png')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Importance\\n",
    "feat_imp = pd.Series(model.feature_importances_, index=X.columns)\\n",
    "feat_imp.sort_values().plot(kind='barh', figsize=(8, 6))\\n",
    "plt.title('Feature Importance')\\n",
    "plt.tight_layout()\\n",
    "plt.savefig('../data/processed/feature_importance.png')\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Save Model ─────────────────────────────────────\\n",
    "joblib.dump(model, '../models/xgboost_model.pkl')\\n",
    "print('Model saved to models/xgboost_model.pkl')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""

PREPROCESS_PY = '''"""
src/preprocess.py
Reusable preprocessing functions.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values — numeric with median, categorical with mode."""
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode all object columns."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(X_train, X_test):
    """Standard scale numeric features."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
'''

TRAIN_PY = '''"""
src/train.py
Model training script — run from command line.
Usage:  python src/train.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from preprocess import load_data, handle_missing, encode_categoricals

DATA_PATH   = "data/raw/YOUR_FILE.csv"   # <-- change this
TARGET_COL  = "target"                    # <-- change this
MODEL_PATH  = "models/xgboost_model.pkl"


def train():
    df = load_data(DATA_PATH)
    df = handle_missing(df)
    df = encode_categoricals(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train()
'''

APP_PY = f'''"""
app.py — Streamlit web app for {PROJECT_NAME}
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="{PROJECT_NAME.replace("-", " ").title()}",
    page_icon="🔬",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_model.pkl")

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────
st.title("🔬 {PROJECT_NAME.replace("-", " ").title()}")
st.markdown("Enter the patient details below and click **Predict** to see the result.")

st.divider()

# ── Input fields (customize per your features) ───────────────────────
col1, col2 = st.columns(2)

with col1:
    age  = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex  = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp   = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

with col2:
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

# Add more inputs as needed ↑

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    # Build input array — match your training feature order
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Result")
    if prediction == 1:
        st.error(f"High Risk  —  Confidence: {{probability*100:.1f}}%")
    else:
        st.success(f"Low Risk  —  Confidence: {{(1-probability)*100:.1f}}%")

st.divider()
st.caption("Built with XGBoost · Streamlit · by [Your Name]")
'''

# ─────────────────────────────────────────────
#  Create everything
# ─────────────────────────────────────────────

def create_structure():
    print(f"\n{'='*50}")
    print(f"  Setting up: {PROJECT_NAME}")
    print(f"{'='*50}\n")

    # 1. Folders
    for folder in FOLDERS:
        os.makedirs(folder, exist_ok=True)
        # Keep folder in git with .gitkeep
        gitkeep = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep):
            open(gitkeep, "w").close()
        print(f"  [OK] Created folder: {folder}/")

    print()

    # 2. Files
    files = {
        ".gitignore":                  GITIGNORE,
        "requirements.txt":            REQUIREMENTS,
        "README.md":                   README,
        "notebooks/01_EDA.ipynb":      EDA_NOTEBOOK,
        "notebooks/02_model.ipynb":    MODEL_NOTEBOOK,
        "src/preprocess.py":           PREPROCESS_PY,
        "src/train.py":                TRAIN_PY,
        "app.py":                      APP_PY,
    }

    for filepath, content in files.items():
        # Don't overwrite existing files
        if os.path.exists(filepath):
            print(f"  [--] Already exists (skipped): {filepath}")
            continue
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  [OK] Created: {filepath}")

    print(f"\n{'='*50}")
    print("  Structure ready!")
    print(f"{'='*50}")
    print("""
  Next steps:
  -----------
  1. Download dataset from Kaggle → data/raw/
  2. Open notebooks/01_EDA.ipynb  → Explore data
  3. Open notebooks/02_model.ipynb → Train model
  4. Run: streamlit run app.py
  5. git add . && git commit -m "Initial project structure" && git push
""")


if __name__ == "__main__":
    create_structure()