import pandas as pd
import sys

from data_download import download_kaggle_dataset
from descriptive_data_analysis import (
    basic_overview, 
    data_quality, 
    univariate_analysis, 
    categorical_analysis, 
    bivariate_analysis, 
    correlation_analysis, 
    statistical_analysis, 
    outlier_detection, 
    full_eda
)
from inferential_data_analysis import (
    inferential_analysis
)

from file_manager import (
    setup_logger
)

from eda import eda
from recommendor import recommend_books



# DOWNLOAD_DATA = True
DOWNLOAD_DATA = False

# EDA = True
EDA = False

RECOMMEND = True
# RECOMMEND = False



def main():
    if DOWNLOAD_DATA :
        download_kaggle_dataset()
    
    df = pd.read_csv("datasets/books.csv")
    
    if EDA :
        # full_eda(df)
        # inferential_analysis(df)
        df = eda(df)

    if RECOMMEND :
        df = pd.read_csv("datasets/books_eda.csv")
        recommend_books(df)




if __name__ == "__main__":
    log_file = setup_logger()
    main()