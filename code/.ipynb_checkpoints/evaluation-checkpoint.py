"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    
    import os
    for file in os.listdir("/opt/ml/processing/model"):
        logger.info(file)
        
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    for file in os.listdir("/opt/ml/processing/model"):
        logger.info(file)
        
    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/"
    
    logger.debug("Reading test data.")
    df = pd.read_csv(test_path+'test.csv', index_col=None, header=None)

    #----------------------------------------------------------
    # TODO - read test data
    # Please fill in this section of code by referring to the reference_notebook.ipynb notebook
    #----------------------------------------------------------

    logger.info("Performing predictions against test data.")
    #----------------------------------------------------------
    # TODO - prediction against test data
    # Please fill in this section of code by referring to the reference_notebook.ipynb notebook
    #----------------------------------------------------------

    # See the regression metrics
    logger.debug("Calculating metrics.")
    mae = mean_absolute_error(y_test, predictions)
    #----------------------------------------------------------
    # TODO - calculate rmse
    # Please fill in this section of code by referring to the reference_notebook.ipynb notebook
    #----------------------------------------------------------
    r2 = r2_score(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mae": {
                "value": mae,
                "standard_deviation": std,
            },
            "mse": {
                "value": mse,
                "standard_deviation": std,
            },
            "rmse": {
                "value": rmse,
                "standard_deviation": std,
            },
            "r2": {
                "value": r2,
                "standard_deviation": std,
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report with rmse: %f", rmse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
