'''
En este scrip tenemos el código que nos saca las predicciones del modelo.
También incluimos integración con MLflow para versionar el modelo
'''
# Librerías

import pandas as pd
import numpy as np
import mlflow
import preprocessors as pp
from helper import BaseLogger

from pipeline import pipeline
import hydra
from hydra.utils import to_absolute_path as abspath
import joblib # Para guardar el modelo
import os
from omegaconf import (DictConfig, OmegaConf)


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, 
                             accuracy_score, 
                             f1_score)
from sklearn.model_selection import train_test_split

logger = BaseLogger()

def load_data(path: DictConfig):
    '''
    Leemos la data para las predicciones, es decir, los test. El path lo tenemos en el 
    archivo yaml de configuracion
    '''
    X_test = pd.read_csv(abspath(path.X_test.path))
    y_test = pd.read_csv(abspath(path.y_test.path))

    return X_test, y_test

def load_model(model_path: str):
    '''
    Cargamos el modelo
    '''
    return joblib.load(model_path)

def predict(model: LogisticRegression, X_test: pd.DataFrame):
    return model.predict(X_test)

def log_params(model: LogisticRegression, features: list):
    logger.log_params({"model_class": type(model).__name__})
    model_params = model.get_params()

    for ag, value in model_params_items():
        logger.log_params({arg: value})
    logger.log_params({"features": features})

def log_metrics(**metrics:dict):
    logger.log_metrics(metrics)

# Función en la que ejecutamos las funciones realizadas arriba
@hydra.main(version_base=None, config_path="../config/", config_name="config")
def evaluate(config: DictConfig):
    mlflow.set_tracking_uri(config.mlflow_tracking_ui)
    mlflow.set_experiment("primer_experimento")

    with mlflow.start_run():
        # Cargamos la data
        X_test, y_test = load_data(config.processed)


        # Cargamos el modelo
        model = load_model(abspath(config.model.path))

        # Obtenemos las predicciones
        yhat = predict(model, X_test)

        # Obtenemos métricas
        f1 = f1_score(y_test, yhat)
        print("###################")
        print(f'El f1 score es de --> {f1}')

        acc = accuracy_score(y_test, yhat)
        print("###################")
        print(f'El accuracy es de --> {acc}')

        # Log metrics en mlflow
        log_params(model, config.process.features)
        log_metrics(f1_score=f1, accuracy_score=acc)
    
        # Aquí tenemos el código para que mlflow nos lo guarde en local
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

if __name__ == "__main__":
    evaluate()


