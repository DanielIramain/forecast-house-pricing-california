import os
import pickle

import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("forecast-pricing-hyperopt")
mlflow.sklearn.autolog(disable=True)

def load_pickle():
    '''Load the pickled data'''
    try:
        with open('../prediction-model/datasets/housing/split_data.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
            
            return loaded_data
    except FileNotFoundError:
        print("Error: The file 'split_data.pkl' was not found.")
    except Exception as e:
        print(f"An error occurred during loading: {e}")

@click.command()
@click.option(
    "--data_path",
    default="./prediction-model/datasets",
    help="Location where the processed California house pricing data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    data = load_pickle()

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "RandomForestRegressor")
            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(data['X_train'], data['y_train'])
            y_pred = rf.predict(data['X_val'])
            rmse = root_mean_squared_error(data['y_val'], y_pred)
            mlflow.log_metric("rmse", rmse)

            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

if __name__ == '__main__':
    run_optimization()