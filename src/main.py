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

@hydra.main(version_base=None, config_path="config/", config_name="preprocessing")
def run(config: DictConfig):
    df = pd.read_csv(config.dataset.data)
    match_pipeline = pipeline(config)
    match_pipeline.fit(df)
    joblib.dump(match_pipeline, utils.to_absolute_path(config.pipeline.pipeline01))

    return df

if __name__ == '__main__':
    run()


    
