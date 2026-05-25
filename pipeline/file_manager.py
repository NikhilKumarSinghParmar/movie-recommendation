import os
import logging
from datetime import datetime


# ==========================================
# CREATE PROJECT DIRECTORIES
# ==========================================

def create_project_folders():

    folders = [
        "logs",
        "plots",
        "outputs",
        "models"
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


# ==========================================
# SETUP LOGGER
# ==========================================

def setup_logger():

    create_project_folders()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    log_file = f"logs/run_{timestamp}.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Logger Initialized")

    return log_file


# ==========================================
# GENERATE PLOT PATH
# ==========================================

def get_plot_path(plot_name):

    return os.path.join(
        "plots",
        f"{plot_name}.png"
    )


# ==========================================
# GENERATE OUTPUT PATH
# ==========================================

def get_output_path(file_name):

    return os.path.join(
        "outputs",
        file_name
    )