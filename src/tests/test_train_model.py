'''
En este script tenemos el código que nos testea el modelo
'''
# Librerías
import joblib
import pandas as pd

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import ModelErrorAnalysis
from hydra import compose, initialize
from hydra.utils import to_absolute_path as abspath

from src.train_model import load_data

def test_reglog():

