
import sys

import os
from pathlib import Path
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from data_utils import calculate_phenoage_df, download_all_needed_files, get_feature_names, get_target_name, load_data
from train import prepare_raw_features


def main():
    years = [2015, 2017]
    
    script_path = Path(os.path.realpath(__file__))
    for year in years:
        data_dir = script_path.parent.parent.absolute() / "data"
        
        # download data
        print("1) downloading data:")
        download_all_needed_files(year, data_dir)

        # load all raw data and calculate target variable
        print("2) loading all data and calculating target")
        data_df = load_data(year, data_dir / str(year))
        data_df = calculate_phenoage_df(data_df)

        # prepare features and target
        print("3) cleaning features and target")
        feature_names = get_feature_names()
        target_name = get_target_name()  
        raw_feature_df = prepare_raw_features(data_df[feature_names+[target_name]])  # target as, features get filtered in the process
        raw_feature_df = raw_feature_df.reset_index(drop=True)

        # saving 
        print("4) saving to parquet file")
        os.makedirs(data_dir / "prepared", exist_ok=True)
        # raw_feature_df.to_csv(data_dir / "prepared" / str(year), index=False, na_rep="NA")
        # read_df = pd.read_csv(data_dir / "prepared" / str(year), dtype={"ALQ130": "Int32"})

        raw_feature_df.to_parquet(data_dir / "prepared" / (str(year)+".parquet"), index=False)
        read_df = pd.read_parquet(data_dir / "prepared" / (str(year)+".parquet"))
        read_df = read_df.map(lambda x: np.nan if x is None else x).astype({"ALQ130": "Int32"})
        
        # print("equal?", raw_feature_df.equals(read_df))
        assert_frame_equal(raw_feature_df, read_df)
        print()

if __name__ == "__main__":
    main()