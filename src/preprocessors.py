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
        columnas = [variable for variable in self.variables]
        matrix = imputer.fit_transform(X_.loc[:, columnas])
        # Agregamos las columnas sin nan al df
        X_[columnas] = matrix
        
        return X_

class StandarNumeric(BaseEstimator, TransformerMixin):
    '''
    Clase para estandarizar las variables métricas
    '''
    def __init__(self, variables):
        self.variables = variables
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = StandardScaler()
        columnas = [variable for variable in self.variables]
        X_ = X.copy()
        matrix_scaled = scaler.fit_transform(X_.loc[:, columnas])
        # Añadimos las columnas escaladas al dataframe
        X_[columnas] = matrix_scaled
        return X_


###########-------- Tratamiento variables categóricas ###########--------

class CatImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        columnas = [variable for variable in self.variables]
        matrix = imputer.fit_transform(X_.loc[:, columnas])
        # Agregamos la matriz con las variables categóricas sin nan
        X_[columnas] = matrix

        return X_

class GetDummie(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        columnas = [variable for variable in self.variables]
        cat = X_.loc[:, columnas]
        cat_dummi = pd.get_dummies(cat)
        # Eliminamos las variables categóricas sin hacer dummies
        X_ = X_.drop(columnas, axis=1)

        # Concatenamos las variables métricas con las categóricas
        X_clean = pd.concat([X_, cat_dummi], axis=1)

        return X_clean

