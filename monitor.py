import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PREDICTION_LOG = Path("monitoring/predictions.csv")
TRAINING_STATS = Path("monitoring/training_stats.json")

PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)


def save_training_stats(df: pd.DataFrame):
    stats = {
        "msg_len_mean": float(df["msg_len"].mean()),
        "msg_len_std": float(df["msg_len"].std()),
        "hour_mean": float(df["hour"].mean()),
        "hour_std": float(df["hour"].std()),
        "row_count": int(len(df)),
    }

    TRAINING_STATS.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATS, "w") as f:
        json.dump(stats, f, indent=2)


def log_prediction(features, prediction):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        **{f"f{i}": v for i, v in enumerate(features)},
        "prediction": prediction,
    }

    df = pd.DataFrame([row])

    if PREDICTION_LOG.exists():
        df.to_csv(PREDICTION_LOG, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG, index=False)


def detect_drift(threshold=3.0):
    if not PREDICTION_LOG.exists() or not TRAINING_STATS.exists():
        print("No monitoring data yet.")
        return False

    preds = pd.read_csv(PREDICTION_LOG)

    with open(TRAINING_STATS) as f:
        baseline = json.load(f)

    current_mean = preds["f0"].mean()
    base_mean = baseline["msg_len_mean"]
    base_std = baseline["msg_len_std"]

    if abs(current_mean - base_mean) > threshold * base_std:
        print("⚠️ DRIFT DETECTED")
        return True

    print("No drift detected.")
    return False
