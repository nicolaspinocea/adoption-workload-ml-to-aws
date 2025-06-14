import json
import pathlib
import pickle
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = pickle.load(open("xgboost-model", "rb"))

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    #y_test = df.iloc[:, 0].to_numpy()

    
        # Intentar convertir la primera columna a datetime si es necesario
    try:
        y_test = pd.to_datetime(df.iloc[:, 0], errors='raise').view(np.int64) / 1e9  # Convierte a segundos
    except Exception as e:
        print(f"Error converting y_test to datetime: {e}")
        # Si ya es numérico, simplemente conviértelo a numpy array
        y_test = df.iloc[:, 0].to_numpy()

    # Elimina la columna de y_test de df
    df.drop(df.columns[0], axis=1, inplace=True)

    # Convertir los valores restantes en X_test y realizar la predicción
    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    #Utilizan metricas de MAPE y MAE
    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
