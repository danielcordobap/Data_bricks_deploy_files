"""Entrena un clasificador binario pequeño y lo guarda como .pkl.

Dataset: Breast Cancer Wisconsin (sklearn), 569 filas, 30 features,
variable respuesta binaria (0 = maligno, 1 = benigno).
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RUTA_SALIDA = Path(__file__).parent / "modelo.pkl"
RUTA_DATOS = Path(__file__).parent / "datos.csv"


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    # Delta/Unity Catalog no admiten espacios en nombres de columna
    X.columns = [c.replace(" ", "_") for c in X.columns]

    # CSV de apoyo para subirlo luego a Databricks como tabla Delta
    df = X.copy()
    df["target"] = y
    df.to_csv(RUTA_DATOS, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    modelo.fit(X_train, y_train)

    proba = modelo.predict_proba(X_test)[:, 1]
    print(f"Accuracy: {accuracy_score(y_test, modelo.predict(X_test)):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.3f}")

    joblib.dump(modelo, RUTA_SALIDA)
    print(f"Modelo guardado en: {RUTA_SALIDA}")
    print(f"Datos guardados en: {RUTA_DATOS}")


if __name__ == "__main__":
    main()
