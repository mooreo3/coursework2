import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("output/output.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "is_same_src_dest" not in df.columns:
        raise ValueError("'is_same_src_dest' not found in dataframe")

    # target as int
    df = df.dropna(subset=["is_same_src_dest"])
    df["is_same_src_dest"] = df["is_same_src_dest"].astype(int)

    return df


# def build_feature_target(df: pd.DataFrame):
#     target_col = "is_same_src_dest"
#
#     # DO NOT USE ANY IP / SUBNET / IP-INT COLUMNS → they leak the target
#     drop_cols = [
#         target_col,
#         # text / high-cardinality / not needed
#         "msg",
#         "timestamp",
#         "date",
#         "time",
#         "block_id",
#         "cls",
#         "path",
#         "node",
#         # IP identity columns
#         "src_ip", "dest_ip", "to_ip", "from_ip",
#         "src_ip_int", "dest_ip_int", "to_ip_int", "from_ip_int",
#         "src_subnet_24", "dest_subnet_24", "to_subnet_24", "from_subnet_24",
#         # old label, irrelevant now
#         "incident",
#     ]
#
#     X = df.drop(columns=drop_cols, errors="ignore")
#     y = df[target_col]
#
#     numeric_features = X.select_dtypes(
#         include=["int64", "int32", "float64", "float32"]
#     ).columns.tolist()
#     categorical_features = [c for c in X.columns if c not in numeric_features]
#
#     print("Using features:")
#     print("  Numeric:", numeric_features)
#     print("  Categorical:", categorical_features)
#
#     return X, y, numeric_features, categorical_features

def build_feature_target(df):
    target_col = "is_same_src_dest"

    # Keep only minimal behavioural features
    keep = ["msg_len", "hour", "minute", "second"]

    X = df[keep]
    y = df[target_col]

    numeric_features = keep
    categorical_features = []  # none

    print("Reduced feature set:")
    print(numeric_features)

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
        class_weight="balanced",  # because 1s are probably rare
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return clf


def main(run_id: str):
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    # Sanity check: class balance
    print("is_same_src_dest distribution:")
    print(df["is_same_src_dest"].value_counts())
    print(df["is_same_src_dest"].value_counts(normalize=True))

    X, y, num_feats, cat_feats = build_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = build_pipeline(num_feats, cat_feats)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    model_path = MODEL_DIR / f"model_{run_id}.pkl"
    metrics_path = MODEL_DIR / f"metrics_{run_id}.json"

    joblib.dump(clf, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "f1": f1}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="local")
    args = parser.parse_args()
    main(run_id=args.run_id)
