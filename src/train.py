
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
from sklearn.metrics import mean_absolute_error
import optuna
import mlflow
from mlflow import MlflowClient

from data_utils import calculate_phenoage_df, download_all_needed_files, get_feature_names, get_target_name, load_data

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment("SVR-hyperparameter-optimization")




def prepare_raw_features(df: pd.DataFrame):
    # TODO new name: preprocess_raw_data_nhanes
    """
    prepare a dataframe with the defined features
    # one-hot categorical features will be transformed to strings for the scikit-learn.DictVectorizer

    Args:
        df (pd.DataFrame): dataframe with the raw features from NHANES data
    """
    df = df.copy() 
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    # df = df.dropna(axis=0)

    # ===== categorical =====
    categorical_one_hot = ["RIAGENDR", "SMQ020"]

    df["RIAGENDR"] = df["RIAGENDR"].astype("Int32")
    df["RIAGENDR"] = df["RIAGENDR"].map({1: "male", 2: "female"})

    df["SMQ020"] = df["SMQ020"].astype("Int32")
    df["SMQ020"] = df["SMQ020"].map(lambda x: {1: "yes", 2: "no"}.get(x, np.nan))  # unknown values become NaN

    # df[categorical_one_hot] = df[categorical_one_hot].astype(str)  # out-commented as this makes nan value to string

    # ===== continuous =====
    # Minutes sedentary activity | only use valid range of 0 to 1320
    # df = df[(df.PAD680 >= 0) & (df.PAD680 <= 1320)]
    df["PAD680"] = df["PAD680"].map(lambda x: x if x >= 0 and x <= 1320 else np.nan)

    # # frequency of alcohol consumption during last 12 months: transform categorical to continuous variable 
    # df["ALQ121"] = df["ALQ121"].astype("Int32")    
    # alcohol_consumption_categorical_to_continuous = {
    #     0: 0.,
    #     1: 365.,  # every day
    #     2: 52.*5,  # almost every day: assume 5 days a week
    #     3: 52.*3.5,  # 3-4 times a week (52 weeks)
    #     4: 52.*2,  # 2 times a week
    #     5: 52.*1,  # once a week
    #     6: 12.*2.5,  # 2-3 times a month
    #     7: 12.*1,  # once a month
    #     8: 9.,  # 9 times a year
    #     9: 4.5,
    #     10: 1.5,
    #     77: np.nan,  # refused
    #     99: np.nan,  # missing
    # }
    # df = df[((df["ALQ121"] >= 0) & (df["ALQ121"] <= 10)) | (df["ALQ121"] == 77) | (df["ALQ121"] == 99)]
    # df["ALQ121"] = df["ALQ121"].replace(alcohol_consumption_categorical_to_continuous)

    df["ALQ130"] = df["ALQ130"].round().astype("Int32")
    df.loc[~df["ALQ130"].between(0, 15, inclusive="both")] = np.nan  # values 777 and 999 are "missing" and "refused"

    df["PAD680"] = df["PAD680"].round()

    df = df.copy().reset_index(drop=True)
    return df


def get_feature_transformer(df_features: pd.DataFrame):
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

    ct = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(drop="if_binary"), categorical_cols),
        ("continuous", StandardScaler(), continuous_cols),
    ])
    ct.fit(df_features)
    return ct


def get_target_transformer(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df[["phenoage"]])
    return scaler


def train_model(X_train, y_train, X_val, y_val, feature_transformer, target_transformer, params: dict):
    # mlflow.sklearn.autolog()
    with mlflow.start_run():

        model = LinearSVR(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        train_error = mean_absolute_error(target_transformer.inverse_transform(y_val.reshape((-1,1))), target_transformer.inverse_transform(y_pred.reshape((-1,1))))
    return model


def hyperparam_tuning(X_train, y_train, X_val, y_val, feature_transformer, target_transformer):

    # TODO use validation error
    def objective(trial: optuna.Trial):
        with mlflow.start_run():
            
            epsilon = trial.suggest_float("epsilon", 0.0, 2.0)
            tol = trial.suggest_float("tol", 1e-5, 1e-3, log=True)
            C = trial.suggest_float("C", 1e-6, 1e2, log=True)
            loss = cast(Literal["epsilon_insensitive", "squared_epsilon_insensitive"], trial.suggest_categorical("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]))
            max_iter = 1000
            random_state = 42

            model = LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss, max_iter=max_iter, random_state=random_state) 
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            val_mean_absolute_error = mean_absolute_error(target_transformer.inverse_transform(y_val.reshape((-1,1))), target_transformer.inverse_transform(y_pred.reshape((-1,1))))

            mlflow.set_tag("model", "LinearSVR")
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"val_mean_absolute_error": val_mean_absolute_error})

            return val_mean_absolute_error 

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    # print(study.best_trial)

    return study.best_trial


def validate_model(X, y):
    pass

def run(year, year_val, batch_val):

    # load data
    script_path = Path(os.path.realpath(__file__))
    data_dir = script_path.parent.parent.absolute() / "data"
    val_dir = script_path.parent.parent.absolute() / "data" 
    
    download_all_needed_files(year, data_dir)
    download_all_needed_files(year_val, val_dir)


    data_df = load_data(year, data_dir / str(year))
    val_df = load_data(year_val, val_dir / str(year_val), batch=batch_val)
    
    # calculate target variable phenoage
    data_df = calculate_phenoage_df(data_df)
    val_df = calculate_phenoage_df(val_df)

    # prepare features and target
    feature_names = get_feature_names()
    target_name = get_target_name()  
    raw_feature_df = prepare_raw_features(data_df[feature_names+[target_name]])  # target as, features get filtered in the process
    feature_transformer = get_feature_transformer(raw_feature_df)
    X_train = feature_transformer.transform(raw_feature_df)

    val_raw_feature_df = prepare_raw_features(val_df[feature_names+[target_name]])  # target as, features get filtered in the process
    X_val = feature_transformer.transform(val_raw_feature_df)


    # prepare target
    target_transformer = get_target_transformer(raw_feature_df)
    y_train = target_transformer.transform(raw_feature_df[[target_name]]).ravel()
    # target_transformer.inverse_transform(y_train.reshape((-1, 1))).min()

    y_val = target_transformer.transform(val_raw_feature_df[[target_name]]).ravel()


    hyperparam_tuning(X_train, y_train, X_val, y_val, feature_transformer, target_transformer)

    # mlflow.set_experiment("best-n-SVR-models")
    # train_model(X_train, y_train, X_val, y_val, feature_transformer, target_transformer)
    # run_id = train_model(preprocessed_features_arr, y_train, feature_transformer, target_transformer)
    # print(f"MLflow run_id: {run_id}")
    # return run_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model for biological age prediction.")
    parser.add_argument("year", type=int, choices=[2015, 2017], help="Year of training data")
    parser.add_argument("year_val", type=int, choices=[2015, 2017], help="Year of validation data")
    parser.add_argument("batch_val", type=int, metavar="[0-9]", choices=range(10), help="Year of validation data")
    args = parser.parse_args()

    run(args.year, args.year_val, args.batch_val)