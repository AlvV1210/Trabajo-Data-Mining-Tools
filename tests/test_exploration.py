import pandas as pd

from data_mining.exploration import dataset_summary, missing_report


def test_dataset_summary_counts_missing_values():
    frame = pd.DataFrame({"edad": [20, None, 30], "ciudad": ["Lima", "Lima", None]})
    summary = dataset_summary(frame).set_index("columna")
    assert summary.loc["edad", "nulos"] == 1
    assert summary.loc["ciudad", "unicos"] == 1


def test_missing_report_orders_by_null_count():
    frame = pd.DataFrame({"a": [1, None, None], "b": [1, 2, None], "c": [1, 2, 3]})
    report = missing_report(frame)
    assert report["columna"].tolist() == ["a", "b"]
