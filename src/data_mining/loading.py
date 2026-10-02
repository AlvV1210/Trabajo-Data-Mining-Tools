"""Lectura y escritura de conjuntos de datos tabulares."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Carga un CSV o Excel según la extensión del archivo."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    raise ValueError(f"Formato no soportado: {suffix}. Usa .csv o .xlsx.")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda un DataFrame como CSV y crea las carpetas necesarias."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination
