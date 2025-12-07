import os
import subprocess

from monitor import detect_drift

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

def main():
    if detect_drift():
        print("Drift detected. Retraining model...")
        subprocess.run(
            [
                "python",
                "train.py",
                "--run-id",
                "auto_retrain",
            ],
            check=True,
        )
        print("Retraining complete.")
    else:
        print("No retraining required.")


if __name__ == "__main__":
    main()
