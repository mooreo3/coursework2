from pathlib import Path
import json
import pandas as pd

from train import main as train_main


def test_train_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    data = pd.DataFrame(
        {
            "msg_len": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "hour":    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            "minute":  [0, 15, 30, 45,  5, 10, 20, 25, 35, 40],
            "second":  [1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
            "is_same_src_dest": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    data_path = tmp_path / "output.csv"
    data.to_csv(data_path, index=False)

    import train as train_module
    monkeypatch.setattr(train_module, "DATA_PATH", data_path)

    model_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "MODEL_DIR", model_dir)

    run_id = "test_run"

    train_main(
        run_id=run_id,
        features=["msg_len", "hour", "minute", "second"],
        n_estimators=10,
        max_depth=5,
    )

    model_path = model_dir / f"model_{run_id}.pkl"
    metrics_path = model_dir / f"metrics_{run_id}.json"

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics
    assert "f1" in metrics
