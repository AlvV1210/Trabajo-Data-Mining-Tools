"""Construcción de variables para modelado."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_numeric(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Estandariza columnas numéricas a media 0 y desviación 1."""
    result = df.copy()
    targets = columns or list(result.select_dtypes(include="number").columns)
    if not targets:
        return result
    scaler = StandardScaler()
    result[targets] = scaler.fit_transform(result[targets])
    return result


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Codifica variables categóricas como variables indicadoras."""
    targets = columns or list(df.select_dtypes(exclude="number").columns)
    if not targets:
        return df.copy()
    return pd.get_dummies(df, columns=list(targets))
