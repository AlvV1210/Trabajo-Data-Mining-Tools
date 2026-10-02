"""Métricas de clasificación y agrupamiento."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, silhouette_score


def classification_scores(y_true, y_pred) -> dict[str, float]:
    """Exactitud y F1 macro."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def clustering_silhouette(features, labels) -> float:
    """Coeficiente de silueta. Requiere al menos dos grupos."""
    return float(silhouette_score(features, labels))
