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
    '''
    En esta función comprobamos que el modelo funciona bien. Los pasos de esta función son:
    1 - Cargamos el modelo ayudándonos de hydra y joblib.
    2- Cargamos los dataset de train y test y concatenamos con las variable target.
    3- Hacemos el check de que el modelo funciona bien y le ponemos un mínimo que tiene que cumplir.
    '''

    with initialize(version_base=None, config_path="../../config"):
        config = compose(config_name="main")

    model_path = abspath(config.model.path)
    model = joblib.load(model_path)

    X_train, X_test, y_train, y_test = load_data(config.processed)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_ds = Dataset(train_df, label="survived")
    validation_ds = Dataset(test_df, label="survived")

    check = ModelErrorAnalysis(min_error_model_score=0.3)
    check.run(train_ds, validation_ds, model)

