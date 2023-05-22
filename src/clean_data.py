'''
En este script tenemos el código de limpieza de datos. Aquí vamos a seleccionar las comlumnas que nos interesa y dividirlas en train y test.
El feature engineering lo tenemos en el de preprocessors.py, que son procesos que incluimos en el pipeline de entrenamiento
'''

# Importamos librerías
import pandas as pd
import numpy as np
import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def get_data(raw_path: str):
    '''
    Leemos la data.
    Input -> el path donde tenemos la data. Ese path lo cogeremos del yaml y usaremos hydra.
    Output -> el dataframe del csv
    '''
    data = pd.read_csv(raw_path)
    return data

def get_features(target: str, features: list, data: pd.DataFrame):
    '''
    Seleccionamos la variable dependiente y las variables independientes.
    Input -> nombre de la columna y, nombres de las X y el dataframe.
    Output -> X e y
    '''
    X = data.loc[:, features]
    y = data.loc[:, target]

    return X, y

@hydra.main(version_base=None, config_path="config/", config_name="config")
def clean(config: DictConfig):
    '''
    Con esta función limpiamos la data
    '''
    data = get_data(config.dataset.data)

    X, y = get_features(target=config.variables.target, features=config.variables.variables, data=data)

    # Dividimos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=7)

    # Save data
    X_train.to_csv(abspath(config.processed.X_train.path), index=False)
    X_test.to_csv(abspath(config.processed.X_test.path), index=False)
    y_train.to_csv(abspath(config.processed.y_train.path), index=False)
    y_test.to_csv(abspath(config.processed.y_test.path), index=False)

if __name__ == "__main__":
    clean()

