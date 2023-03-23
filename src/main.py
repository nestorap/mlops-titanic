'''
En este script tenemos el proceso principal
'''
import pandas as pd
import preprocessors as pp

from pipeline import pipeline
import hydra
from hydra import utils
import joblib
import os
from omegaconf import DictConfig, OmegaConf

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

@hydra.main(version_base=None, config_path="config/", config_name="preprocessing")
def run(config: DictConfig):
    df = pd.read_csv(config.dataset.data)
    #X = df.loc[:, config.variables.variables]
    #y = df.loc[:,"survived"]

    # Separamos en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        df.loc[:, config.variables.variables], # Seleccionamos variables independientes
        df.loc[:, config.variables.target], # seleccionamos variable dependiente
        test_size=0.2,
        random_state=0
    )
    # Ejecutamos el pipeline
    match_pipeline = pipeline(config)
    match_pipeline.fit(X_train, y_train)
    joblib.dump(match_pipeline, utils.to_absolute_path(config.pipeline.pipeline01))
    yhat = match_pipeline.predict(X_test)
    print(f'confusion matrix --> \ {confusion_matrix(y_test, yhat)}')
    #df.to_csv("../data/processed/titanic_preprocess.csv")

    return df

if __name__ == '__main__':
    run()


    
