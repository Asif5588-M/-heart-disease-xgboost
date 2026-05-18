# Heart Disease Xgboost

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

## Problem Statement
Heart disease is one of the leading causes of death worldwide. 
This project predicts whether a patient has heart disease based 
on 13 clinical features using XGBoost classifier.

## Dataset
- **Source:** Kaggle — [Heart Disease UCI](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- **Rows:** 303 patients
- **Features:** 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target:** 1 = Heart Disease, 0 = No Disease

## Model Used
## Model Used
| Model | Accuracy | ROC-AUC | Recall (Disease) |
|-------|----------|---------|-----------------|
| XGBoost | 81.97% | 87.66% | 97% |

## Key Insights
- **thal** and **cp** are the most important features
- Model achieves **97% recall** for Heart Disease detection — critical for medical use
- Balanced dataset: 165 positive, 138 negative cases

## Project Structure
```
heart-disease-xgboost/
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
git clone https://github.com/YOUR_USERNAME/heart-disease-xgboost.git
cd heart-disease-xgboost

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
Asif Malik — [GitHub](https://github.com/Asif5588-M) | [Upwork Profile](your-upwork-link)
