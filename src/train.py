"""
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
