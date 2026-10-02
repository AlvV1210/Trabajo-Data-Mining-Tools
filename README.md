# Trabajo Data Mining Tools

Herramientas en Python para un flujo de data mining: cargar datos, limpiarlos, explorarlos, construir variables, entrenar modelos y evaluar resultados.

## Estructura

```text
.
├── config/                 # Parámetros de ejemplo del flujo
├── data/
│   ├── raw/                # Datos originales, sin modificar
│   ├── interim/            # Transformaciones intermedias
│   ├── processed/          # Datos listos para modelar
│   └── external/           # Fuentes de terceros
├── notebooks/              # Análisis exploratorio en Jupyter
├── src/data_mining/        # Librería reutilizable
├── scripts/                # Flujos ejecutables de punta a punta
├── tests/                  # Pruebas de las herramientas
├── models/                 # Modelos entrenados (no se versionan)
└── reports/figures/        # Gráficos generados (no se versionan)
```

Los archivos de `data/`, `models/` y `reports/figures/` quedan fuera de Git. Solo se conserva la carpeta para que la estructura viaje con el repositorio.

## Instalación

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Para cuadernos Jupyter:

```bash
pip install -e ".[notebooks]"
```

## Uso rápido

```python
from data_mining.loading import load_table
from data_mining.preprocessing import drop_duplicates, impute_missing
from data_mining.exploration import dataset_summary

datos = load_table("data/raw/dataset.csv")
datos = impute_missing(drop_duplicates(datos))
print(dataset_summary(datos))
```

Un recorrido con datos sintéticos:

```bash
py -3 scripts/ejemplo_flujo.py
```

## Pruebas

```bash
py -3 -m pytest
```
