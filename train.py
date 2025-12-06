from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib


DATA_PATH = Path("output/output.csv")   # adjust if your CSV lives somewhere else
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    # low_memory=False to avoid the mixed dtypes warning
    df = pd.read_csv(path, low_memory=False)

    if "incident" not in df.columns:
        raise ValueError(
            f"'incident' column not found in {path}. "
            f"Make sure you're using the output from your preprocess script."
        )

    # Drop rows where label is missing
    df = df.dropna(subset=["incident"])

    # Make sure incident is int (0/1)
    df["incident"] = df["incident"].astype(int)

    return df


def build_feature_target(df: pd.DataFrame):
    # Columns we absolutely don't want as features
    drop_cols = [
        "incident",      # label
        "date",
        "time",
        "msg",
        "timestamp",
        "block_id",
        "src_ip",
        "dest_ip",
        "to_ip",
        "from_ip",
        "src_subnet_24",
        "dest_subnet_24",
        "to_subnet_24",
        "from_subnet_24",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["incident"]

    # Infer numeric vs categorical
    numeric_features = X.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return clf


def main():
    df = load_data(DATA_PATH)
    X, y, numeric_features, categorical_features = build_feature_target(df)

    print(f"Loaded {len(df)} rows")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline(numeric_features, categorical_features)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # Save model pipeline (preprocessing + model together)
    model_path = MODEL_DIR / "model_v1.pkl"
    joblib.dump(clf, model_path)

    # Save metrics
    metrics_path = MODEL_DIR / "metrics_v1.json"
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1": f1}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
