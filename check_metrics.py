import json
from pathlib import Path
import sys

THRESHOLDS = {
    "accuracy": 0.99,
    "f1": 0.99,
}

def main(run_id: str = "local"):
    metrics_path = Path("models") / f"metrics_{run_id}.json"
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    acc = metrics.get("accuracy")
    f1 = metrics.get("f1")

    print(f"Loaded metrics: accuracy={acc}, f1={f1}")
    ok = True

    if acc is None or f1 is None:
        print("Missing metrics in JSON")
        ok = False
    if acc is not None and acc < THRESHOLDS["accuracy"]:
        print(f"FAIL: accuracy {acc} < {THRESHOLDS['accuracy']}")
        ok = False
    if f1 is not None and f1 < THRESHOLDS["f1"]:
        print(f"FAIL: f1 {f1} < {THRESHOLDS['f1']}")
        ok = False

    if not ok:
        sys.exit(1)

    print("Metrics thresholds satisfied.")


if __name__ == "__main__":
    run_id = "local"
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    main(run_id)
