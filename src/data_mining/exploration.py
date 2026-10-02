"""Resúmenes para el análisis exploratorio."""

from __future__ import annotations

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con tipo, nulos y valores únicos por columna."""
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "columna": column,
                "tipo": str(series.dtype),
                "nulos": int(series.isna().sum()),
                "pct_nulos": float(series.isna().mean() * 100),
                "unicos": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas con valores faltantes, de mayor a menor cantidad."""
    summary = dataset_summary(df)
    missing = summary.loc[summary["nulos"] > 0]
    return missing.sort_values("nulos", ascending=False).reset_index(drop=True)


def numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos de las columnas numéricas."""
    return df.select_dtypes(include="number").describe().T
