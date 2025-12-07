from pathlib import Path
import pandas as pd

from preprocess import run


def test_preprocess_run_tmp_file(tmp_path: Path):
    log_path = tmp_path / "sample.log"
    out_csv = tmp_path / "output.csv"

    lines = [
        "240101 000001 10 INFO org.apache.hadoop.hdfs.DataNode: Received block blk_-1 of size 4096 from /10.0.0.1\n",
        "240101 000002 11 INFO org.apache.hadoop.hdfs.DataNode: Receiving block blk_-2 src: /10.0.0.2:50010 dest: /10.0.0.3:50020\n",
        "240101 000003 12 ERROR org.apache.hadoop.hdfs.DataNode: Served block blk_-3 to /10.0.0.4\n",
    ]
    log_path.write_text("".join(lines), encoding="utf-8")

    df = run(input_path=str(log_path), output_dataframe_path=str(out_csv))

    assert not df.empty

    for col in ["timestamp", "hour", "minute", "second", "msg_len", "incident"]:
        assert col in df.columns

    assert out_csv.exists()

    df2 = pd.read_csv(out_csv)
    assert len(df2) == len(df)

    error_row = df[df["level"].str.upper() == "ERROR"].iloc[0]
    assert error_row["incident"] == 1
