'''
En este script tenemos el proceso principal
'''
import hydra
from clean_data import clean
from train import train
from evaluate_model import evaluate




#@hydra.main(version_base=None, config_path="config/", config_name="config")
#def run(config: DictConfig):
#    df = pd.read_csv(config.dataset.data)

    # Separamos en train y test
#    X_train, X_test, y_train, y_test = train_test_split(
#        df.loc[:, config.variables.variables], # Seleccionamos variables independientes
#        df.loc[:, config.variables.target], # seleccionamos variable dependiente
#        test_size=0.2,
#        random_state=0
#    )
    # Ejecutamos el pipeline
#    match_pipeline = pipeline(config)
#    match_pipeline.fit(X_train, y_train)
#    joblib.dump(match_pipeline, utils.to_absolute_path(config.pipeline.pipeline01))
#    yhat = match_pipeline.predict(X_test)
#    print(f'confusion matrix --> \ {confusion_matrix(y_test, yhat)}')

#   return df
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(config):
    #clean(config)
    #train(config)
    evaluate(config)


if __name__ == '__main__':
    main()
