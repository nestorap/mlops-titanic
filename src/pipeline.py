'''
En este srcipt tenmemos el pipeline de preprocesamiento
'''
# Librerías
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
import preprocessors as pp
import hydra

def pipeline(config):
    match_pipeline = Pipeline(
        [
            (
                'numerical_imputer',
                pp.NumericalImputer(variables=config.variables.variables_numeric)
            ),
            (
                'numerical_standar',
                pp.StandarNumeric(variables=config.variables.variables_numeric)
            ),
            (
                'categorical_imputer',
                pp.CatImputer(variables=config.variables.variables_cat)
            ),
            (
                'categorical_dummies',
                pp.GetDummie(variables=config.variables.variables_cat)
            )
        ]
    )
    return match_pipeline
