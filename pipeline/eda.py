import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns





def basic_info(df):
    print("Shape of the dataset:", df.shape)
    print(f"\nsize : {df.size}")
    print(f"\nColumns : {df.columns}")
    print("\nData types of each column:\n", df.dtypes)

    print(f"\n\n head :\n {df.head()}")
    print(f"\n\ntail : \n{df.tail()}")

    print("\nDescription of numerical columns:\n", df.describe())
    print(f"\n info : {df.info()}")





def missing_analysis(df):
    missing = df.isna().sum()
    missing_percent = (missing / len(df) * 100).round(3)
    

    missing_df = pd.DataFrame({
        "missing_values" : missing,
        "missing_percent" : missing_percent
    })

    
    missing_order = missing_df[missing_df["missing_values"] > 0].sort_values(by="missing_percent", ascending=False)
    
    print("Missing Values:")
    print(missing)
    
    print("\nMissing Values (%):")
    print(missing_percent)

    print(f"\n missing order : \n{missing_order}")

    return missing_df, missing_order




def duplicate_analysis(df):
    duplicates = df.duplicated().sum()
    duplicates_percent = (duplicates / len(df)) * 100


    duplicates_df = pd.DataFrame({
        "duplicates": [duplicates],
        "duplicates_percent": [duplicates_percent]
    })

    duplicates_order = duplicates_df[duplicates_df["duplicates"] > 0].sort_values(by="duplicates_percent", ascending=False)

    print("\n\n Duplicate Rows : \n", duplicates)
    print("\n\n Duplicate Rows (%) : \n", duplicates_percent)

    print(f"\n\n duplicates order : \n{duplicates_order}")


    return duplicates_df, duplicates_order



def column_types(df):
    num_cols = df.select_dtypes(
        include = ["int64","float64"]
    )
    cat_cols = df.select_dtypes(
        include = ["object"]
    )


    print(f"\n\n Numerical Columns : \n{num_cols.columns}")
    print(f"\n\n Categorical Columns : \n{cat_cols.columns}")   

    return num_cols, cat_cols




def handle_isbn(df):
    df["isbn10"] = df["isbn10"].astype("int64", errors="ignore")
    df['isbn13'] = df['isbn13'].astype("int64", errors="ignore")

    return df




def handle_meta_data(df):
    df["title"] = df["title"].fillna("Unknown")
    
    df["subtitle"] = np.where(
        df["subtitle"].isna(),
        df["title"],
        df["subtitle"]
    )
    
    df["description"] = np.where(
        df["description"].isna(),
        df['title'] + " " +  df['subtitle'],
        df["description"]
    )

    df["meta_data"] = (
        df["title"] + " " +
        df["subtitle"] + " " +
        df["authors"] + " " +
        df["categories"] + " " +
        df["description"]
    )

    df["meta_data"] = (
        df["meta_data"]
        .str.replace(f"\s+", " ", regex = True)
    )

    print(f"type of the meta_data : {type(df["meta_data"].iloc[0])}")
    print(f" meta data head : {df["meta_data"].iloc[0]}")

    return df



def eda(df):
    basic_info(df)
    _,_ = missing_analysis(df)
    _,_ = duplicate_analysis(df)
    _,_ = column_types(df)
    df = handle_isbn(df)
    df = handle_meta_data(df)

    df.to_csv(
        "datasets/books_eda.csv",
        index = False
    )

    return df





if __name__ == "__main__" :
    df = pd.read_csv("datasets/books.csv")
    basic_info(df)
    _,_ = missing_analysis(df)
    _,_ = duplicate_analysis(df)
    _,_ = column_types(df)
    df = handle_isbn(df)
    df = handle_meta_data(df)

    basic_info(df)