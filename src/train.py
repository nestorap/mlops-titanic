'''
En este script vamos a tener la parte de entrenamiento. Va a consistir en leer los distintos datos y entrenar el modelo
ejecutando el pipeline y el preprocessors.py para guardar un modelo de ml
'''
# Importamos librerías
import pandas as pd
import numpy as np

import preprocessors as pp
from pipeline import pipeline
import hydra
from hydra.utils import to_absolute_path as abspath
import os
from omegaconf import (DictConfig, OmegaConf)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix)
import joblib # Para guardar el modelo

def load_data_train(path: DictConfig):
    '''
    Leemos los distintos datos
    '''
    X_train = pd.read_csv(abspath(path.X_train.path))
    y_train = pd.read_csv(abspath(path.y_train.path))

    return X_train, y_train


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(config: DictConfig):
    '''
    Función de entrenamiento. Cogemos los datos ya separados pero sin limpiar
    y le pasamos por el pipeline. Al final, guardamos el modelo
    '''
    # Leemos la data
    X_train, y_train = load_data_train(config.processed)
    
    # Instanciamos el pipeline
    match_pipeline = pipeline(config)
    # Entrenamos y obtenemos un modelo con el pipeline
    match_pipeline.fit(X_train, y_train)
    yhat_train = match_pipeline.predict(X_train)
    # Sacamos métricas de train
    print("#### Métricas de train ####")
    acc_train = accuracy_score(yhat_train, y_train)
    print(f'Accuracy de train es --> {acc_train}')
    print(" ##### CONFUSION MATRIX ###### \n", confusion_matrix(yhat_train, y_train))
    
    # Guardamos el modelo
    joblib.dump(match_pipeline, abspath(config.model.path))    

if __name__ == "__main__":
    train()
