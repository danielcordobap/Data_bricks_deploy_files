# Data_bricks_deploy_files

Companion files for a step-by-step tutorial on **deploying a machine learning model on Databricks** using only the free tier (Databricks Free Edition): from a local `.pkl` file to a Unity Catalog model served as a REST endpoint, with a Feature Store and scheduled retraining.

The model itself is deliberately simple. A small classifier trained on the Breast Cancer Wisconsin dataset (569 rows, 30 numeric features, binary target) is used so the focus stays on the **deployment lifecycle**, not on model quality.

## Repository contents

| File | What it is |
|---|---|
| [`train_model.py`](train_model.py) | Trains the model locally and writes `modelo.pkl` and `datos.csv`. A scikit-learn `Pipeline` (`StandardScaler` + `RandomForestClassifier`). It also renames the columns (`mean radius` → `mean_radius`) because Delta tables do not allow spaces in column names. |
| [`modelo.pkl`](modelo.pkl) | The trained model, serialized with `joblib`. This is the artifact that gets deployed. It was saved with scikit-learn 1.9.0, so install the same version when loading it. |
| [`datos.csv`](datos.csv) | The training data (569 rows, 30 features plus `target`), ready to be uploaded to Databricks and turned into a Delta table. |
| [`Ml_model_cancer.ipynb`](Ml_model_cancer.ipynb) | The Databricks notebook with the whole workflow, cell by cell (see below). |

## What the notebook does

`Ml_model_cancer.ipynb` is meant to run in a Databricks notebook on **serverless** compute. It covers:

1. **Data:** creates a `workspace.ml` schema in Unity Catalog and turns `datos.csv` into the Delta table `workspace.ml.breast_cancer`.
2. **Load and test:** loads `modelo.pkl` inside Databricks and checks its predictions against the table.
3. **Register with MLflow:** logs the model with its signature and an input example, and registers it in Unity Catalog as `workspace.ml.cancer_classifier`.
4. **Alias:** assigns the `champion` alias to version 1, so consumers can follow an alias instead of a hard-coded version.
5. **Serve:** a manual UI step to create a Model Serving endpoint (`cancer-classifier`), followed by a code cell that calls it.
6. **Feature Store:** splits features and labels, creates the `cancer_features` feature table with `patient_id` as primary key, builds a training set with lookups, and registers a second model (`cancer_classifier_fs`) linked to those features.

Some notebook paths point to a workspace folder (`/Workspace/deploy_ml_model/...`). Adjust them to wherever you uploaded `modelo.pkl` and `datos.csv`.

## How to run it

1. Train the model locally, or use the provided files:
   ```bash
   pip install scikit-learn pandas joblib
   python train_model.py
   ```
2. Create a [Databricks Free Edition](https://www.databricks.com/learn/free-edition) account and complete the identity verification.
3. In your Databricks workspace, create a folder and upload `modelo.pkl` and `datos.csv`.
4. Import `Ml_model_cancer.ipynb` (or paste its cells into a new notebook), attach **Serverless** compute and run the cells in order. Cells that start with `%pip` restart Python, so run them on their own.

## The blog

A written tutorial accompanies these files. It explains each step with screenshots and covers more than the notebook does:

- Creating the workspace and uploading the files.
- Registering the model with MLflow and giving it an alias.
- Creating the Model Serving endpoint and calling it, both from the UI and from outside Databricks with a token (PowerShell, curl and Python).
- Building the Feature Store.
- A scheduled Databricks Job for **continuous retraining**: it trains a candidate, registers a new version, compares it with the current `champion` and promotes it by moving the alias.
- A table of common errors and how to fix them.

The blog is available in English and Spanish.

## Notes and limitations

- The accuracy logged during registration is computed on the training data, so it says nothing about real quality. Always evaluate on held-out data in a real project.
- The retraining job's champion-versus-candidate comparison is only fair if neither model has seen the test rows. This is documented in the blog.
- A serving endpoint serves a specific model version, not an alias, so moving `champion` does not update the endpoint by itself.
- Serving a model with real-time feature lookup needs an online table, which may not be available in Free Edition.
- Never commit access tokens or credentials. Calling the endpoint from outside Databricks requires a token: keep it in an environment variable and revoke it when you are done.

## Tech stack

Databricks (Free Edition, serverless) · Unity Catalog · Delta Lake · MLflow · Databricks Feature Engineering · Model Serving · scikit-learn · Python
