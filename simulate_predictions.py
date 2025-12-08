from pathlib import Path

import joblib
import pandas as pd

from monitor import log_prediction

MODEL_DIR = Path("models")
RUN_ID = "demo_msg_plus_time"
MODEL_PATH = MODEL_DIR / f"model_{RUN_ID}.pkl"

#
def main():
    clf = joblib.load(MODEL_PATH)

    samples = pd.DataFrame(
        [
            [5000, 1, 1, 1],
            [8000, 2, 2, 2],
            [12000, 3, 3, 3],
            [15000, 4, 4, 4],
        ],
        columns=["msg_len", "hour", "minute", "second"],
    )

    preds = clf.predict(samples)

    for features, pred in zip(samples.values.tolist(), preds):
        print(f"Features={features} -> pred={pred}")
        log_prediction(features, int(pred))


if __name__ == "__main__":
    main()
