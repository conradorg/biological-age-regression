
import sys

import os
from pathlib import Path
import numpy as np
import pandas as pd 
from pandas.testing import assert_frame_equal

from data_utils import calculate_phenoage_df, download_all_needed_files, get_feature_names, get_target_name, load_data, load_preprocessed_data_parquet, preprocess_raw_data_nhanes


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
        columns = ["SEQN"] + get_feature_names() + [get_target_name()]
        raw_feature_df = preprocess_raw_data_nhanes(data_df[columns])
        raw_feature_df = raw_feature_df.reset_index(drop=True)

        # saving 
        print("4) saving to parquet file")
        os.makedirs(data_dir / "prepared", exist_ok=True)
        # raw_feature_df.to_csv(data_dir / "prepared" / str(year), index=False, na_rep="NA")
        # read_df = pd.read_csv(data_dir / "prepared" / str(year), dtype={"ALQ130": "Int32"})

        outfile_path = data_dir / "prepared" / f"{year}.parquet"
        raw_feature_df.to_parquet(outfile_path, index=False)
        read_df = load_preprocessed_data_parquet(outfile_path)
        
        # print("equal?", raw_feature_df.equals(read_df))
        assert_frame_equal(raw_feature_df, read_df)
        print("-----")

if __name__ == "__main__":
    main()