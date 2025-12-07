import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import mlflow
import mlflow.sklearn

from monitor import save_training_stats


DATA_PATH = Path("output/output.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if "is_same_src_dest" not in df.columns:
        raise ValueError("'is_same_src_dest' not found in dataframe")

    df = df.dropna(subset=["is_same_src_dest"])
    df["is_same_src_dest"] = df["is_same_src_dest"].astype(int)

    return df


def build_feature_target(df: pd.DataFrame, feature_list: List[str]):
    """
    Select features and target from the dataframe.

    feature_list comes from CLI (e.g. ["msg_len", "hour", "minute", "second"])
    so we can easily run multiple experiments with different feature sets.
    """
    target_col = "is_same_src_dest"

    keep = feature_list

    missing = [f for f in keep if f not in df.columns]
    if missing:
        raise ValueError(f"Requested features not in dataframe columns: {missing}")

    X = df[keep]
    y = df[target_col]

    numeric_features = keep
    categorical_features: List[str] = []

    print("Using feature set:")
    print(numeric_features)

    return X, y, numeric_features, categorical_features


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    n_estimators: int,
    max_depth: Optional[int],
):
    """
    Build sklearn Pipeline with preprocessing + RandomForest.

    n_estimators and max_depth are passed in so we can vary them between runs.
    """
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
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return clf, model


def main(
    run_id: str,
    features: List[str],
    n_estimators: int,
    max_depth: Optional[int],
):
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("is_same_src_dest distribution:")
    print(df["is_same_src_dest"].value_counts())
    print(df["is_same_src_dest"].value_counts(normalize=True))

    save_training_stats(df)

    X, y, num_feats, cat_feats = build_feature_target(df, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf, base_model = build_pipeline(num_feats, cat_feats, n_estimators, max_depth)

    mlflow.set_experiment("self_loop_detection_v2")

    with mlflow.start_run(run_name=f"run_{run_id}"):
        mlflow.log_params(
            {
                "n_estimators": base_model.n_estimators,
                "max_depth": base_model.max_depth,
                "class_weight": base_model.class_weight,
                "test_size": 0.2,
                "random_state": 42,
                "features": ",".join(num_feats),
            }
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 score: {f1:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODEL_DIR / f"model_{run_id}.pkl"
        metrics_path = MODEL_DIR / f"metrics_{run_id}.json"

        joblib.dump(clf, model_path)
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc, "f1": f1}, f, indent=2)

        print(f"Saved model to {model_path}")
        print(f"Saved metrics to {metrics_path}")

        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        mlflow_model_dir = MODEL_DIR / f"mlflow_model_{run_id}"
        mlflow_model_dir.mkdir(parents=True, exist_ok=True)

        mlflow.sklearn.save_model(
            sk_model=clf,
            path=str(mlflow_model_dir),
        )

        mlflow.log_artifacts(str(mlflow_model_dir), artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="local")

    parser.add_argument(
        "--features",
        default="msg_len,hour,minute,second",
        help="Comma-separated list of feature names to use",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in the RandomForest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Max depth of trees in the RandomForest (None = unlimited)",
    )

    args = parser.parse_args()

    feature_list = [f.strip() for f in args.features.split(",") if f.strip()]

    main(
        run_id=args.run_id,
        features=feature_list,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
