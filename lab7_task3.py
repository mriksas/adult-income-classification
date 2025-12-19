import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_FILE = "adult.data"
TEST_FILE = "adult.test"  # optional

OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_adult_dataset(train_path: str, test_path: str | None = None) -> pd.DataFrame:
    df_train = pd.read_csv(
        train_path, header=None, names=COLUMNS, sep=", ", engine="python"
    )
    dfs = [df_train]

    if test_path and os.path.exists(test_path):
        df_test = pd.read_csv(
            test_path, header=None, names=COLUMNS, sep=", ", engine="python", skiprows=1
        )
        # sometimes test labels have trailing dot
        df_test["income"] = df_test["income"].astype(str).str.replace(".", "", regex=False)
        dfs.append(df_test)

    return pd.concat(dfs, ignore_index=True)


def build_preprocessor(categorical_features, numeric_features, binary_features):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
            ("bin", "passthrough", binary_features),
        ],
        remainder="drop"
    )
    return preprocessor


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
    }


def save_confusion_matrix_plot(cm, title, out_path):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def main():
    ensure_dirs()

    df = load_adult_dataset(TRAIN_FILE, TEST_FILE if os.path.exists(TEST_FILE) else None)

    # ✅ IMPORTANT FIX: use np.nan, not pd.NA
    df = df.replace("?", np.nan)

    # Strip whitespace safely for object columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            # revert "nan" strings back to real NaN (because astype(str) turns NaN into "nan")
            df.loc[df[col].str.lower() == "nan", col] = np.nan

    # Target
    df["income"] = df["income"].astype(str).str.replace(".", "", regex=False)
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df.dropna(subset=["income"])
    df["income"] = df["income"].astype(int)

    # Binary encode sex
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    categorical = [
        "workclass", "education", "marital_status",
        "occupation", "relationship", "race", "native_country"
    ]
    numeric = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    binary = ["sex"]

    X = df[categorical + numeric + binary]
    y = df["income"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(categorical, numeric, binary)

    models = {
        "Logistic Regression (baseline)": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest (tree-based)": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_split=4, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting (advanced)": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    results = []
    details = {}

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results.append({"Model": name, **metrics})
        details[name] = {
            "hyperparameters": pipe.named_steps["model"].get_params(),
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        }

        safe = "".join(ch if ch.isalnum() or ch in " _-" else "_" for ch in name)
        save_confusion_matrix_plot(cm, f"Confusion Matrix — {name}", os.path.join(PLOTS_DIR, f"cm_{safe}.png"))

        print("\n" + "=" * 80)
        print(name)
        print(metrics)
        print("CM:\n", cm)

    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "model_details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    print(results_df)
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
