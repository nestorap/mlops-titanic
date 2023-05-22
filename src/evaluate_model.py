'''
En este scrip tenemos el código que nos saca las predicciones del modelo
'''
# Librerías

import pandas as pd
import numpy as np
import preprocessors as pp

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

# Función en la que ejecutamos las funciones realizadas arriba
@hydra.main(version_base=None, config_path="../config/", config_name="config")
def evaluate(config: DictConfig):
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

if __name__ == "__main__":
    evaluate()


