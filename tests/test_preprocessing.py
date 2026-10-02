import pandas as pd

from data_mining.preprocessing import drop_duplicates, impute_missing, remove_outliers_iqr


def test_drop_duplicates_removes_repeated_rows():
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    result = drop_duplicates(frame)
    assert len(result) == 2
    assert result.iloc[0].tolist() == [1, "x"]


def test_impute_missing_fills_numeric_with_median():
    frame = pd.DataFrame({"valor": [1.0, None, 3.0]})
    result = impute_missing(frame, strategy="median")
    assert result["valor"].tolist() == [1.0, 2.0, 3.0]


def test_impute_missing_rejects_unknown_strategy():
    frame = pd.DataFrame({"valor": [1.0]})
    try:
        impute_missing(frame, strategy="zero")
    except ValueError:
        return
    raise AssertionError("debía rechazar la estrategia")


def test_remove_outliers_iqr_drops_extreme_value():
    frame = pd.DataFrame({"valor": [10, 11, 12, 13, 100]})
    result = remove_outliers_iqr(frame, columns=["valor"])
    assert 100 not in result["valor"].tolist()
    assert len(result) == 4
