import os
import shutil
from kagglehub import dataset_download
from dotenv import load_dotenv

load_dotenv()

KAGGLE_DATA_PATH = os.getenv("KAGGLE_DATA_PATH")
TARGET_DATA_PATH = os.getenv("TARGET_DATA_PATH")


def download_kaggle_dataset():
    data_path = dataset_download(
        KAGGLE_DATA_PATH
    )

    print(f"Dataset downloaded to : {data_path}")

    source_path = os.path.join(
        data_path,
        "books.csv"
    )


    destination_path = os.path.join(
        TARGET_DATA_PATH,
        "books.csv"
    )


    os.makedirs(
        TARGET_DATA_PATH,
        exist_ok=True
    )

    if os.path.exists(destination_path):

        os.remove(destination_path)

    shutil.copy(
        source_path,
        destination_path
    )

    print(f"books.csv copied to : {destination_path}")


if __name__ == "__main__":

    download_kaggle_dataset()
    

# from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# api.authenticate()

# api.dataset_download_files(
#     "dylanjcastillo/7k-books-with-metadata",
#     path="datasets",
#     unzip=True
# )