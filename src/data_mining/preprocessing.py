"""Limpieza y preparación de conjuntos de datos."""

from __future__ import annotations

import pandas as pd


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia sin filas duplicadas."""
    return df.drop_duplicates().reset_index(drop=True)


def impute_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Imputa valores faltantes.

    En columnas numéricas, ``strategy`` puede ser ``median``, ``mean`` o ``mode``.
    Las columnas no numéricas se completan con la moda.
    """
    if strategy not in {"median", "mean", "mode"}:
        raise ValueError("strategy debe ser 'median', 'mean' o 'mode'.")

    result = df.copy()
    for column in result.select_dtypes(include="number").columns:
        series = result[column]
        if strategy == "mean":
            fill = series.mean()
        elif strategy == "mode":
            modes = series.mode(dropna=True)
            fill = modes.iloc[0] if not modes.empty else 0
        else:
            fill = series.median()
        result[column] = series.fillna(fill)

    for column in result.select_dtypes(exclude="number").columns:
        modes = result[column].mode(dropna=True)
        if not modes.empty:
            result[column] = result[column].fillna(modes.iloc[0])
    return result


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Elimina filas con valores atípicos según el rango intercuartílico."""
    result = df.copy()
    targets = columns or list(result.select_dtypes(include="number").columns)
    mask = pd.Series(True, index=result.index)
    for column in targets:
        series = result[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask &= series.between(lower, upper) | series.isna()
    return result.loc[mask].reset_index(drop=True)
