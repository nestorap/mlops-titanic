'''
En este script tenemos el código que testea el preprocesamiento de los datos.
'''

# Librerías
import pandas as pd
from pandera import (Check, Column, DataFrameSchema)
from pytest_steps import test_steps

from src.clean_data import get_features # Importamos la función que queremos testear.

# Iniciamos el test con el decorator y haciendo referencia a la función que queremos testear
@test_steps("get_features_step")
def test_process_suite(test_step, steps_data):
    get_features_step(steps_data)

def get_features_step(steps_data):
    data = pd.DataFrame(
        {
            "age" : [0.0, 90.0],
            "fare" : [0.0, 513.0],
            "pclass" : [1, 2, 3],
            "sex" : ["male", "female"],
            "embarked" : ["S", "C", "Q"],
            "class" : ["First", "Second", "Third"],
            "who" : ["man", "woman", "child"],
            "survived" : [0, 1]
        }
    )

    features = [
        "age",
        "fare",
        "sex"
    ]

    target = "survived"

    y, X = get_features(target, features, data)

    schema = DataFrameSchema(
        {
            "age" : Column(float, Check.greater_than_or_equal_to(0)),
            "fare" : Column(float, Check.greater_than_or_equal_to(0)),
            "sex[male]" : Column(float, Check.isin([0.0, 1.0])),
            "sex[male]" : Column(float, Check.isin([0.0, 1.0]))
        }
    )

    schema.validate(X)
