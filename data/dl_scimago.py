import io
import os
import json
import requests
import argparse
import pandas as pd
from tqdm import tqdm

# SCIMAGO Journal Rank url
_BASE_URL = "https://www.scimagojr.com/journalrank.php"

def parse_args():
    parser = argparse.ArgumentParser(description="Download and parse scimago SJRs and H indexes from specified year range")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./scimago/",
        help="path to the output dir where scimago files will be stored, if not defined, files will be stored at ./scimago/"
    )
    parser.add_argument(
        "--start_year",
        type=str,
        default=1999,
        help="year (included) from which scimago data will be downloaded"
    )
    parser.add_argument(
        "--end_year",
        type=str,
        default=2022,
        help="last year (included) to which scimago data will be downloaded"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    sjr_global_dict = {}
    hind_global_dict = {}

    for year in tqdm(range(args.start_year, args.end_year+1)):
        ## DOWNLOAD REQUEST
        params = {
            'year': str(year),
            'out': 'xls',
        }
        try :
            response = requests.get(_BASE_URL, params=params)
            if response.status_code == 200:
                csv_file = io.StringIO(response.text)
            else:
                print(f"Failed to download file for year {year}. Status code: {response.status_code}.")
        except Exception as e:
            print(f"An error occurred while making the request for year {year}: {str(e)}")
        ## LOAD DATA
        df = pd.read_csv(csv_file, sep=';', decimal=",", usecols=["Issn","SJR","H index"]) # decimals use commas in SJR column
        ## PROCESS
        df["SJR"] = df["SJR"].fillna(0.0) # replace NaN values in SJR column
        df.drop(df[df["Issn"] == "-"].index, inplace=True) # remove journals without ISSN number
        df["Issn"] = df["Issn"].str.split(", ") # split journal with multiple issns into lists
        df = df.explode("Issn") # explode journal with multiple Issn on multiple lines (one per ISSN)
        df.drop_duplicates(subset="Issn",keep="first", inplace=True) # drop duplicate journals
        df.set_index("Issn", inplace=True) # set unique ISSN column as index
        ## OUTPUT
        sjr_dict = df["SJR"].to_dict()
        # sjr
        if not sjr_global_dict :
            sjr_global_dict = {issn:{year:sjr} for issn,sjr in sjr_dict.items()}
        else :
            for issn,sjr in sjr_dict.items() :
                if issn not in sjr_global_dict:
                    sjr_global_dict[issn] = {year:sjr}
                else :
                    sjr_global_dict[issn][year] = sjr
        # h index
        hind_dict = df["H index"].to_dict()
        if not hind_global_dict:
            hind_global_dict = hind_dict
        else :
            for issn,hind in hind_dict.items():
                 if issn not in hind_global_dict:
                    hind_global_dict[issn] = hind
        
    with open(os.path.join(args.output_path,"issn_sjr.json"),"w") as f:
        json.dump(sjr_global_dict,f)

    with open(os.path.join(args.output_path,"issn_h-index.json"),"w") as f:
        json.dump(hind_global_dict,f)

if __name__ == "__main__":
    main()