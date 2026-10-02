"""Gráficos de apoyo al análisis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_missing_bars(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda un gráfico de barras con el porcentaje de nulos por columna."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    missing = df.isna().mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="#2f6f8f")
    ax.set_ylabel("Porcentaje de nulos")
    ax.set_xlabel("Columna")
    ax.set_title("Valores faltantes")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


def save_correlation_heatmap(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda un mapa de calor con la correlación de las columnas numéricas."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, cmap="Blues", ax=ax)
    ax.set_title("Correlación")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination
