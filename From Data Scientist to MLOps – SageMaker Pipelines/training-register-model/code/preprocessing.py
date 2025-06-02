import os
import sys
import subprocess
import re
import calendar
import logging
import warnings
warnings.filterwarnings('ignore')
# Actualiza pip
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
# Instalar databricks-sql-connector y pandas
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pandas"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "packaging"])

from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

if __name__ == "__main__":
    BASE_DIR = "/opt/ml/processing/input"
    INPUT_FILES = [os.path.join(f'{BASE_DIR}', file)
                   for file in os.listdir(f'{BASE_DIR}')]
    
    print("files script: ", INPUT_FILES)
    file_to_remove = f'{BASE_DIR}/input/code'

    # Verificar si el archivo está en la lista y eliminarlo solo si existe
    if file_to_remove in INPUT_FILES:
        INPUT_FILES.remove(file_to_remove)

    FILE_PATH= os.path.join(f'{BASE_DIR}', "data-original.csv")

    df = pd.read_csv(FILE_PATH, sep = ",")
    df_clean = df.copy()

    train_size = int(0.7 * len(df_clean))
    val_size = int(0.85 * len(df_clean))

    train, validation, test = np.split(df_clean, [train_size, val_size])

    pd.DataFrame(train).to_csv(f"{BASE_DIR}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(f"{BASE_DIR}/validation/validation.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(f"{BASE_DIR}/test/test.csv", header=False, index=False)
