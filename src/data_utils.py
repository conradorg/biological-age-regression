import requests
import os
import numpy as np
import pandas as pd

biomarker2fileprefix = {
    "LBDSALSI": "BIOPRO",  # g/L
    "LBDSCRSI": "BIOPRO",  # umol/L
    "LBDSGLSI": "BIOPRO",  # mmol/L
    "LBXHSCRP": "HSCRP",  # mg/L
    "LBXLYPCT": "CBC",  # %
    "LBXMCVSI": "CBC",  # fL
    "LBXRDW": "CBC",  # %
    "LBXSAPSI": "BIOPRO",  # U/L
    "LBXWBCSI": "CBC",  # 1000 cells/uL
    "RIDAGEYR": "DEMO",  # Years
}
biomarker2description = {
    "LBDSALSI": "Albumin, refrigerated serum(g/L)",
    "LBDSCRSI": "Creatinine, refrigerated serum (umol/L)",
    "LBDSGLSI": "Glucose, refrigerated serum (mmol/L)",  # mmol/L
    "LBXHSCRP": "High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)",  # TODO: mg/L -> log(mg/dL) (for phenoage calculation)
    "LBXLYPCT": "Lymphocyte percent (%)",  # %
    "LBXMCVSI": "Mean cell volume (fL)",  # fL
    "LBXRDW": "Red cell distribution width (%)",  # %
    "LBXSAPSI": "Alkaline Phosphatase (ALP) (IU/L)",  # IU/L == U/L
    "LBXWBCSI": "White blood cell count (1000 cells/uL)",  # 1000 cells/uL
    "RIDAGEYR": "Age in years of the participant at the time of screening. Individuals 80 and over are topcoded at 80 years of age.",  # Years
    "LBXHSCRP_mg_dL": "[self-calculated] High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)",  # mg/dL
}
feature2description = {
    "RIDAGEYR": "Age in years of the participant at the time of screening. Individuals 80 and over are topcoded at 80 years of age.",
    "RIAGENDR": "Gender of the participant.",
    "BMXHT": "Standing Height (cm)",
    "BMXWAIST": "Waist Circumference (cm)",
    "BMXWT": "Weight (kg)",
    "PAD680": "(Minutes) The following question is about sitting at school, at home, getting to and from places, or with friends including time spent sitting at a desk, traveling in a car or bus, reading, playing cards, watching television, or using a computer. Do not include time spent sleeping. How much time {do you/does SP} usually spend sitting on a typical day?",
    "SMQ020": "These next questions are about cigarette smoking and other tobacco use. {Have you/Has SP} smoked at least 100 cigarettes in {your/his/her} entire life?",
    # "ALQ121": "ALQ121 - Past 12 mo how often have alcohol drink",
    "ALQ130": "ALQ130 - Avg # alcohol drinks/day - past 12 mos",
    "SLD012": "Number of hours usually sleep on weekdays or workdays.",
    # "SLD013": "Number of hours usually sleep on weekends or non-workdays.",
}
feature2fileprefix = {
    "RIDAGEYR": "DEMO",
    "RIAGENDR": "DEMO",
    "BMXHT": "BMX",
    "BMXWAIST": "BMX",
    "BMXWT": "BMX",
    "PAD680": "PAQ",
    "SMQ020": "SMQ",
    # "ALQ121": "ALQ",
    "ALQ130": "ALQ",
    "SLD012": "SLQ",
    # "SLD013": "SLQ",
}

# Variable search: https://wwwn.cdc.gov/nchs/nhanes/search/default.aspx
col2description = {
    "LBDSALSI": "Albumin, refrigerated serum(g/L)",
    "LBDSCRSI": "Creatinine, refrigerated serum (umol/L)",
    "LBDSGLSI": "Glucose, refrigerated serum (mmol/L)",  # mmol/L
    "LBXHSCRP": "High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)",  # TODO: mg/L -> log(mg/dL) (for phenoage calculation)
    "LBXLYPCT": "Lymphocyte percent (%)",  # %
    "LBXMCVSI": "Mean cell volume (fL)",  # fL
    "LBXRDW": "Red cell distribution width (%)",  # %
    "LBXSAPSI": "Alkaline Phosphatase (ALP) (IU/L)",  # IU/L == U/L
    "LBXWBCSI": "White blood cell count (1000 cells/uL)",  # 1000 cells/uL
    "RIDAGEYR": "Age in years of the participant at the time of screening. Individuals 80 and over are topcoded at 80 years of age.",  # Years
    "LBXHSCRP_mg_dL": "[self-calculated] High-Sensitivity C-Reactive Protein (hs-CRP) (mg/L)",  # mg/dL
}
feature_names = [
    # "LBDSALSI",
    # "LBDSCRSI",
    # "LBDSGLSI",
    # "LBXHSCRP_mg_dL",  # instead of LBXHSCRP!
    # "LBXLYPCT",
    # "LBXMCVSI",
    # "LBXRDW",
    # "LBXSAPSI",
    # "LBXWBCSI",
    # "RIDAGEYR",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXHT",
    "BMXWAIST",
    "BMXWT",
    "PAD680",
    "SMQ020",
    # "ALQ121",
    "ALQ130",
    "SLD012",
    # "SLD013",
]


def download_file(url, filename):
    response = requests.get(url)
    if not response.ok:
        raise ValueError(f"Requested file not available at url {url}")
    with open(filename, mode="wb") as file:
        file.write(response.content)


def get_nhanes_suffix(year):
    """get NHANES suffix for a specific start year"""
    suffixes = {
        1999: "A",
        2001: "B",
        2003: "C",
        2005: "D",
        2007: "E",
        2009: "F",
        2011: "G",
        2013: "H",
        2015: "I",
        2017: "J",
        2019: "K",
        2021: "L",
    }
    if year in suffixes:
        return suffixes[year]
    else:
        raise ValueError(f"no NHANES suffix known for year {year}")


def get_nhanes_url(year, file_prefix):
    return (
        f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{str(year)}/DataFiles/{get_nhanes_filename(year, file_prefix)}"
    )


def get_nhanes_filename(year, file_prefix):
    return f"{file_prefix}_{get_nhanes_suffix(year)}.xpt"


def download_all_needed_files(year, data_dir):
    unique_file_prefixes = np.unique(list((biomarker2fileprefix | feature2fileprefix).values()))
    data_path = os.path.join(data_dir, str(year))
    for file_prefix in unique_file_prefixes:
        os.makedirs(data_path, exist_ok=True)
        file_url = get_nhanes_url(year, file_prefix)
        download_file(file_url, os.path.join(data_path, get_nhanes_filename(year, file_prefix)))
        print(file_url, os.path.join(data_path, get_nhanes_filename(year, file_prefix)))


def load_data(year, data_dir):
    # try to merge based on SEQN/patient
    unique_file_prefixes = np.unique(list((biomarker2fileprefix | feature2fileprefix).values()))
    files_to_read = [
        os.path.join(data_dir, get_nhanes_filename(year, file_prefix)) for file_prefix in unique_file_prefixes
    ]
    raw_dfs = [pd.read_sas(f) for f in files_to_read]

    result_df = raw_dfs[0]
    for raw_df in raw_dfs[1:]:
        result_df = result_df.merge(raw_df, on="SEQN")
    # result_df = result_df.filter((biomarker2fileprefix|feature2fileprefix).keys()).copy()

    return result_df


def log_robust(x: np.ndarray, epsilon: float = 1e-6):
    """
    robust natural logarithm to prevent numerical instability at log(0) which is not defined (negative infinity)
    it replaces values x smaller than epsilon with epsilon. epsilon is a small constant.

    Args:
        x (np.ndarray): values x for which the natural logarithm is calculated
        epsilon (float): small constant

    Returns:
        np.ndarray: natural log of max(x, epsilon)
    """
    return np.log(np.maximum(x, epsilon))


def calculate_phenoage_numpy(
    albumin: np.ndarray,
    creatinine: np.ndarray,
    glucose: np.ndarray,
    c_reactive_protein: np.ndarray,
    lymphocyte_percent: np.ndarray,
    mean_cell_volume: np.ndarray,
    red_cell_distribution_width: np.ndarray,
    alkaline_phosphatase: np.ndarray,
    white_blood_cell_count: np.ndarray,
    age: np.ndarray,
):
    """
    calculation of phenoage using biomarkers, reference: https://www.aging-us.com/article/101414/text with supplementary material
    all biomarkers are provided as arrays. The function returns the phenoage in a resulting array of the same shape as all input arrary.

    Args:
        albumin (np.ndarray): Albumin, refrigerated serum(g/L)
        creatinine (np.ndarray): Creatinine, refrigerated serum (umol/L)
        glucose (np.ndarray): Glucose, refrigerated serum (mmol/L)
        c_reactive_protein (np.ndarray): High-Sensitivity C-Reactive Protein (hs-CRP) (mg/dL)
        lymphocyte_percent (np.ndarray): Lymphocyte percent (%)
        mean_cell_volume (np.ndarray): Mean cell volume (fL)
        red_cell_distribution_width (np.ndarray): Red cell distribution width (%)
        alkaline_phosphatase (np.ndarray): Alkaline Phosphatase (ALP) (IU/L)
        white_blood_cell_count (np.ndarray): White blood cell count (1000 cells/uL)
        age (np.ndarray): Age in years of the participant at the time of screening. Individuals 80 and over are topcoded at 80 years of age.

    Returns:
        np.ndarray: calculated array of phenoage
    """
    xb = (
        -19.9067
        - 0.0336 * albumin
        + 0.0095 * creatinine
        + 0.1953 * glucose
        + 0.0954 * log_robust(c_reactive_protein)
        - 0.0120 * lymphocyte_percent
        + 0.0268 * mean_cell_volume
        + 0.3306 * red_cell_distribution_width
        + 0.0019 * alkaline_phosphatase
        + 0.0554 * white_blood_cell_count
        + 0.0804 * age
    )

    mortality_score = 1 - np.exp(-np.exp(xb) * ((np.exp(120 * 0.0076927) - 1) / 0.0076927))
    with np.errstate(divide="ignore"):
        phenoage = 141.50225 + np.log(-0.00553 * np.log(1 - mortality_score)) / 0.090165

    return phenoage


def calculate_phenoage_df(df):
    df = df.copy()
    df["LBXHSCRP_mg_dL"] = df["LBXHSCRP"] * 10
    df.loc[:, ["phenoage"]] = calculate_phenoage_numpy(
        albumin=df["LBDSALSI"].to_numpy(),
        creatinine=df["LBDSCRSI"].to_numpy(),
        glucose=df["LBDSGLSI"].to_numpy(),
        c_reactive_protein=df["LBXHSCRP_mg_dL"].to_numpy(),
        lymphocyte_percent=df["LBXLYPCT"].to_numpy(),
        mean_cell_volume=df["LBXMCVSI"].to_numpy(),
        red_cell_distribution_width=df["LBXRDW"].to_numpy(),
        alkaline_phosphatase=df["LBXSAPSI"].to_numpy(),
        white_blood_cell_count=df["LBXWBCSI"].to_numpy(),
        age=df["RIDAGEYR"].to_numpy(),
    )
    return df


def get_feature_names():
    return feature_names


def get_target_name():
    return "phenoage"


def preprocess_raw_data_nhanes(df: pd.DataFrame):
    """
    prepare a dataframe with the defined features

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


def load_preprocessed_data_parquet(filepath, batch: int | None = None, num_batches=5, dropna=True):
    read_df = pd.read_parquet(filepath)
    read_df = read_df.map(lambda x: np.nan if x is None else x).astype({"ALQ130": "Int32"})
    if dropna:
        read_df = read_df.dropna(axis=0)  # drop rows containing missing values
    if batch is not None:

        if batch not in range(num_batches):
            raise ValueError("batch must be None or in range(num_batches)")
        batch_size = len(read_df) // num_batches
        start = batch * batch_size
        end = (batch + 1) * batch_size if batch < num_batches - 1 else len(read_df)

    return read_df
