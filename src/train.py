import sys
from typing import Literal, cast

sys.path.append("..")


import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from data_utils import (
    get_feature_names,
    get_target_name,
    load_preprocessed_data_parquet,
)

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"  # set timeout for MLflow HTTP requests
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "3"  # set maximum retries for MLflow HTTP requests

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
print("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
print("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def convert_str_values_to_float(d):
    new_dict = {}
    for k, v in d.items():
        try:
            new_dict[k] = float(v)
        except (ValueError, TypeError):
            new_dict[k] = v
    return new_dict


def create_X(df_preprocessed: pd.DataFrame, feature_transform: ColumnTransformer | None = None):
    df = df_preprocessed[get_feature_names()].copy()
    if feature_transform is None:
        feature_transform = get_feature_transform(df)
    return feature_transform.transform(df), feature_transform


def create_y(df_preprocessed: pd.DataFrame, target_transform: StandardScaler | None = None):
    df = df_preprocessed[[get_target_name()]].copy()
    if target_transform is None:
        target_transform = get_target_transform(df)
    return target_transform.transform(df).ravel(), target_transform


def get_feature_transform(df_features: pd.DataFrame):
    categorical_cols = ["RIAGENDR", "SMQ020"]
    continuous_cols = [
        "RIDAGEYR",
        "BMXHT",
        "BMXWAIST",
        "BMXWT",
        "PAD680",
        # "ALQ121",
        "ALQ130",
        "SLD012",
        # "SLD013",
    ]

    ct = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="if_binary"), categorical_cols),
            ("continuous", StandardScaler(), continuous_cols),
        ]
    )
    ct.fit(df_features)
    return ct


def get_target_transform(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df[[get_target_name()]])
    return scaler


def train_model(X_train, y_train, X_val, y_val, feature_transform, target_transform, params: dict):
    inverse_ttf = lambda y: target_transform.inverse_transform(y.reshape((-1, 1)))

    # mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        model = LinearSVR(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        train_mean_absolute_error = mean_absolute_error(inverse_ttf(y_train), inverse_ttf(y_pred))
        train_mean_squared_error = mean_squared_error(inverse_ttf(y_train), inverse_ttf(y_pred))

        y_pred = model.predict(X_val)
        val_mean_absolute_error = mean_absolute_error(inverse_ttf(y_val), inverse_ttf(y_pred))
        val_mean_squared_error = mean_squared_error(inverse_ttf(y_val), inverse_ttf(y_pred))

        mlflow.set_tag("model", "LinearSVR")
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "train_mean_absolute_error": train_mean_absolute_error,
                "train_mean_squared_error": train_mean_squared_error,
                "val_mean_absolute_error": val_mean_absolute_error,
                "val_mean_squared_error": val_mean_squared_error,
            }
        )

        mlflow.sklearn.log_model(model, name="model")
        mlflow.sklearn.log_model(feature_transform, name="feature_transform")
        mlflow.sklearn.log_model(target_transform, name="target_transform")

    return run_id, model


def hyperparam_tuning(X_train, y_train, X_val, y_val, feature_transformer, target_transform):
    inverse_ttf = lambda y: target_transform.inverse_transform(y.reshape((-1, 1)))

    def objective(trial: optuna.Trial):
        with mlflow.start_run():

            epsilon = trial.suggest_float("epsilon", 0.0, 2.0)
            tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)
            C = trial.suggest_float("C", 1e-6, 1e2, log=True)
            loss = cast(
                Literal["epsilon_insensitive", "squared_epsilon_insensitive"],
                trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
            )
            max_iter = 1000
            random_state = 42

            model = LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss, max_iter=max_iter, random_state=random_state)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_train)
            train_mean_absolute_error = mean_absolute_error(inverse_ttf(y_train), inverse_ttf(y_pred))
            train_mean_squared_error = mean_squared_error(inverse_ttf(y_train), inverse_ttf(y_pred))

            y_pred = model.predict(X_val)
            val_mean_absolute_error = mean_absolute_error(inverse_ttf(y_val), inverse_ttf(y_pred))
            val_mean_squared_error = mean_squared_error(inverse_ttf(y_val), inverse_ttf(y_pred))

            mlflow.set_tag("model", "LinearSVR")
            mlflow.log_params(trial.params)
            mlflow.log_metrics(
                {
                    "train_mean_absolute_error": train_mean_absolute_error,
                    "train_mean_squared_error": train_mean_squared_error,
                    "val_mean_absolute_error": val_mean_absolute_error,
                    "val_mean_squared_error": val_mean_squared_error,
                }
            )

            return val_mean_absolute_error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    # print(study.best_trial)

    return study.best_trial


def run_hyperparameter_tuning(year: int, year_val: int, batch_val: int):

    mlflow.set_experiment("SVR-hyperparameter-optimization")

    # load data
    print("load data")
    script_path = Path(os.path.realpath(__file__))
    data_dir = script_path.parent.parent.absolute() / "data"
    df_train = load_preprocessed_data_parquet(data_dir / "prepared" / f"{year}.parquet")
    df_val = load_preprocessed_data_parquet(data_dir / "prepared" / f"{year_val}.parquet", batch=batch_val)

    # prepare data
    print("prepare data")
    X_train, feature_transform = create_X(df_train)
    y_train, target_transform = create_y(df_train)

    X_val, _ = create_X(df_val, feature_transform=feature_transform)
    y_val, _ = create_y(df_val, target_transform=target_transform)

    # hyperparameter tuning
    print("start hyperparameter tuning")
    hyperparam_tuning(X_train, y_train, X_val, y_val, feature_transform, target_transform)


def run_save_training_best_3(year, year_val, batch_val):

    mlflow.set_experiment("best-n-SVR-models")

    # load data
    print("load data")
    script_path = Path(os.path.realpath(__file__))
    data_dir = script_path.parent.parent.absolute() / "data"
    df_train = load_preprocessed_data_parquet(data_dir / "prepared" / f"{year}.parquet")
    df_val = load_preprocessed_data_parquet(data_dir / "prepared" / f"{year_val}.parquet", batch=batch_val)

    # prepare data
    print("prepare data")
    X_train, feature_transform = create_X(df_train)
    y_train, target_transform = create_y(df_train)

    X_val, _ = create_X(df_val, feature_transform=feature_transform)
    y_val, _ = create_y(df_val, target_transform=target_transform)

    # look up best hyperparameter from hyperparam tuning experiment
    hyperparam_exp = "SVR-hyperparameter-optimization"
    print(f"look up mlflow for best runs in experiment >{hyperparam_exp}<")
    hyperparam_experiment = client.get_experiment_by_name(hyperparam_exp)
    if hyperparam_experiment is None:
        raise ValueError(f"experiment >{hyperparam_experiment}< does not exist but is needed")
    best_runs = client.search_runs(
        experiment_ids=[hyperparam_experiment.experiment_id],
        order_by=["metrics.val_mean_absolute_error ASC"],  # models with smallest error first
        max_results=3,
    )

    for run in best_runs:
        print(run.info.artifact_uri)
        print("===== Start training =====")
        print("run_id:", run.info.run_id)
        print("params:", run.data.params)
        print("metrics:", run.data.metrics)

        params = convert_str_values_to_float(run.data.params)
        train_model(X_train, y_train, X_val, y_val, feature_transform, target_transform, params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model for biological age prediction.")
    parser.add_argument("year", type=int, choices=[2015, 2017], help="Year of training data")
    parser.add_argument("year_val", type=int, choices=[2015, 2017], help="Year of validation data")
    parser.add_argument("batch_val", type=int, metavar="[0-4]", choices=range(5), help="Year of validation data")
    parser.add_argument(
        "--hyperparam_tune",
        type=str,
        default="True",
        help="if True hyperparameter tuning, else train best models from hyperparameter tuning",
    )
    args = parser.parse_args()

    if args.hyperparam_tune == "True":
        run_hyperparameter_tuning(args.year, args.year_val, args.batch_val)
    else:
        run_save_training_best_3(args.year, args.year_val, args.batch_val)
