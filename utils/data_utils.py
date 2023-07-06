from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import time
from sklearn.model_selection import GridSearchCV
from typing import Dict, Union, List


def run_model(model, params: Dict[str, List[float]],  X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame):

    grid_search = GridSearchCV(model, params, cv=5)


    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    fit_time = end_time - start_time

    start_time = time.time()
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    end_time = time.time()
    predict_time = end_time - start_time

    print(f"Best Parameters: {best_model}")
    return model, y_pred, fit_time, predict_time

def get_results_classification(model,
                model_name: str,params:Dict[str, Union[float,int]],
                X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame):

    fitted_model, y_pred, fit_time, predict_time = run_model(model, params, X_train, X_test, y_train, y_test)
    acc: float =  accuracy_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred, average='macro')
    print(f"{model_name} Accuracy: {acc}")
    print(f"{model_name} f1: {f1}")
    print(f"{model_name} confusion matrix: {confusion_matrix(y_test, y_pred) }")
    print(f"{model_name} Fit Time: {fit_time} seconds")
    print(f"{model_name} Predict Time: {predict_time} seconds")
    return f1

def get_results_linear(model, 
                model_name: str,
                X_train: pd.DataFrame, X_test: pd.DataFrame, 
                y_train: pd.DataFrame, y_test: pd.DataFrame):
    
    fitted_model, y_pred, fit_time, predict_time = run_model(model, X_train, X_test, y_train, y_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model_name} RMSE: {rmse}")
    print(f"{model_name} R2 Score: {r2}")
    print(f"{model_name} MAE: {mae}\n")

    return rmse, r2, mae