"""Entrenamiento de modelos habituales en data mining."""

from __future__ import annotations

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def split_train_test(features, target, test_size: float = 0.2, random_state: int = 42):
    """Divide características y objetivo en entrenamiento y prueba."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def train_random_forest(features, target, n_estimators: int = 100, random_state: int = 42):
    """Ajusta un bosque aleatorio para clasificación."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(features, target)
    return model


def train_kmeans(features, n_clusters: int = 3, random_state: int = 42):
    """Agrupa observaciones con K-Means."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(features)
    return model
