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
from hydra import utils
import os
from omegaconf import (DictConfig, OmegaConf)

from sklearn.linear_model import LogisticRegression
import joblib # Para guardar el modelo

def load_data(path: DictConfig):
    '''
    Leemos los distintos datos
    '''
    X_train = pd.read_csv(abspath(path.X_train.path))
    X_test = pd.read_csv(abspath(path.X_test.path)
    y_train = pd.read_csv(abspath(path.y_train.path)
    y_test = pd.read_csv(abspath((path.y_test.path))

    return X_train, X_test, y_train, y_test


