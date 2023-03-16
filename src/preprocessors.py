'''
En este script tengo als clases de preprocesamiento.
La idea es usar hydra para la automatización del proceso y usar pipelines
'''
# Importamos librerías
import pandas as pd
import numpy as np

import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.base import (BaseEstimator, TransformerMixin)


###########-------- Tratamiento variables métricas ###########--------
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        matrix = imputer.fit_transform(X_.loc[:, variables])
        # Agregamos las columnas sin nan al df
        X_[variables] = matrix
        return X_

class StandarNumeric(BaseEstimator, TransformerMixin):
    '''
    Clase para estandarizar las variables métricas
    '''
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y=None):
        return self

    def scalado(self, X):
        scaler = StandardScaler()
        X_ = X.copy()
        matrix_scaled = scaler.fit_transform(X_.loc[:,variables])
        # Añadimos las columnas escaladas al dataframe
        X_[variables] = matrix_scaled
        return X_


###########-------- Tratamiento variables categóricas ###########--------

class CatImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return fit

    def cat_imputer(self, X):
        X_ = X.copy()
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        matrix = imputer.fit_transform(X_.loc[:,variables])
        # Agregamos la matriz con las variables categóricas sin nan
        X_[variables] = matrix

        return X_

class GetDummie(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return fit

    def dummies(self, X):
        X_ = X.copy()
        cat = X_.loc[:,variables]
        cat_dummi = pd.get_dummies(cat)
        # Eliminamos las variables categóricas sin hacer dummies
        X_ = X_drop(variables, axis=1)

        # Concatenamos las variables métricas con las categóricas
        X_clean = pd.concat([X_, cat_dummi], axis=1)

        return X_clean

