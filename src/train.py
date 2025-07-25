
import sys
sys.path.append("..")


import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_absolute_error

from data_utils import calculate_phenoage_df, get_feature_names, get_target_name, load_data


def prepare_raw_features(df: pd.DataFrame):
    """
    prepare a dataframe with the defined features
    # one-hot categorical features will be transformed to strings for the scikit-learn.DictVectorizer

    Args:
        df (pd.DataFrame): dataframe with the raw features from NHANES data
    """
    df = df.copy() 
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    df = df.dropna(axis=0)

    # ===== categorical =====
    categorical_one_hot = ["RIAGENDR", "SMQ020"]

    df["RIAGENDR"] = df["RIAGENDR"].astype(np.int32)
    df["RIAGENDR"] = df["RIAGENDR"].replace({1: "male", 2: "female"})

    df["SMQ020"] = df["SMQ020"].astype(np.int32)
    df["SMQ020"] = df["SMQ020"].map(lambda x: {1: "yes", 2: "no"}.get(x, np.nan))  # unknown values become NaN

    df[categorical_one_hot] = df[categorical_one_hot].astype(str)

    # ===== continuous =====
    # Minutes sedentary activity | only use valid range of 0 to 1320
    df = df[(df.PAD680 >= 0) & (df.PAD680 <= 1320)]

    # frequency of alcohol consumption during last 12 months: transform categorical to continuous variable 
    df["ALQ121"] = df["ALQ121"].astype(np.int32)
    
    alcohol_consumption_categorical_to_continuous = {
        0: 0.,
        1: 365.,  # every day
        2: 52.*5,  # almost every day: assume 5 days a week
        3: 52.*3.5,  # 3-4 times a week (52 weeks)
        4: 52.*2,  # 2 times a week
        5: 52.*1,  # once a week
        6: 12.*2.5,  # 2-3 times a month
        7: 12.*1,  # once a month
        8: 9.,  # 9 times a year
        9: 4.5,
        10: 1.5,
        77: np.nan,  # refused
        99: np.nan,  # missing
    }
    df = df[((df["ALQ121"] >= 0) & (df["ALQ121"] <= 10)) | (df["ALQ121"] == 77) | (df["ALQ121"] == 99)]
    df["ALQ121"] = df["ALQ121"].replace(alcohol_consumption_categorical_to_continuous)

    df["PAD680"] = df["PAD680"].astype(np.int32)

    df = df.dropna(axis=0).reset_index()
    return df


def get_feature_transformer(df_features: pd.DataFrame):
    categorical_cols = ["RIAGENDR", "SMQ020"]
    continuous_cols = [
        "RIDAGEYR",
        "BMXHT",
        "BMXWAIST",
        "BMXWT",
        "PAD680",
        "ALQ121",
        "SLD012",
        "SLD013",
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


def train_model(X, y, feature_transformer, target_transformer):
    model = LinearSVR(C=0.1)
    model.fit(X, y)
    y_pred = model.predict(X)

    train_error = mean_absolute_error(target_transformer.inverse_transform(y.reshape((-1,1))), target_transformer.inverse_transform(y_pred.reshape((-1,1))))
    print("train_error", train_error)
    return model


def validate_model(X, y):
    pass

def run(year):

    # load data
    script_path = Path(os.path.realpath(__file__))
    data_dir = script_path.parent.parent.absolute() / "data" / str(year)
    data_df = load_data(year, data_dir)
    
    # calculate target variable phenoage
    data_df = calculate_phenoage_df(data_df)

    # prepare features and target
    feature_names = get_feature_names()
    target_name = get_target_name()  
    raw_feature_df = prepare_raw_features(data_df[feature_names+[target_name]])  # target as, features get filtered in the process
    feature_transformer = get_feature_transformer(raw_feature_df)
    preprocessed_features_arr = feature_transformer.transform(raw_feature_df)

    # prepare target
    target_transformer = get_target_transformer(raw_feature_df)
    preprocessed_target_arr = target_transformer.transform(raw_feature_df[[target_name]]).ravel()
    # target_transformer.inverse_transform(preprocessed_target_arr.reshape((-1, 1))).min()

    train_model(preprocessed_features_arr, preprocessed_target_arr, feature_transformer, target_transformer)
    # run_id = train_model(preprocessed_features_arr, preprocessed_target_arr, feature_transformer, target_transformer)
    # print(f"MLflow run_id: {run_id}")
    # return run_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model for biological age prediction.")
    parser.add_argument("--year", type=int, required=True, choices=[2015, 2017], help="Year of training data")
    args = parser.parse_args()

    run(args.year)