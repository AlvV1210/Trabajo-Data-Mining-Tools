"""Recorre el flujo de data mining con un conjunto sintético."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_mining.evaluation import classification_scores
from data_mining.exploration import dataset_summary
from data_mining.modeling import split_train_test, train_random_forest
from data_mining.preprocessing import drop_duplicates, impute_missing
from data_mining.visualization import save_correlation_heatmap


def build_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 120
    ingreso = rng.normal(2500, 600, size)
    edad = rng.integers(18, 70, size)
    clase = (ingreso > 2600).astype(int)
    frame = pd.DataFrame({"edad": edad, "ingreso": ingreso, "clase": clase})
    frame.loc[0, "ingreso"] = np.nan
    frame.loc[1] = frame.loc[2]
    return frame


def main() -> None:
    data = impute_missing(drop_duplicates(build_sample()))
    print(dataset_summary(data).to_string(index=False))

    features = data[["edad", "ingreso"]]
    target = data["clase"]
    x_train, x_test, y_train, y_test = split_train_test(features, target)
    model = train_random_forest(x_train, y_train)
    scores = classification_scores(y_test, model.predict(x_test))
    print(scores)

    figure = save_correlation_heatmap(
        data,
        ROOT / "reports" / "figures" / "correlacion_ejemplo.png",
    )
    print(f"Figura guardada en {figure}")


if __name__ == "__main__":
    main()
