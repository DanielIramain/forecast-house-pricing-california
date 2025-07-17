import os
import pickle
import click
import mlflow

from hpo import load_pickle

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

'''
Will check the results from the previous experiment and select the top 5 runs.
After that, it will calculate the RMSE of those models on the val set 
And save the results to a new experiment called random-forest-pricing-best-models
'''

HPO_EXPERIMENT_NAME = "forecast-pricing-hyperopt"
EXPERIMENT_NAME = "random-forest-pricing-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def train_and_log_model(data_path, params):

    data = load_pickle()

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(data['X_train'], data['y_train'])

        # Evaluate model on the validation set
        val_rmse = root_mean_squared_error(data['y_val'], rf.predict(data['X_val']))
        mlflow.log_metric("val_rmse", val_rmse)

        # Log the model and return the run ID
        mlflow.sklearn.log_model(rf, "model")

@click.command()
@click.option(
    "--data_path",
    default="./prediction-model/datasets",
    help="Location where the processed California house pricing data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    run_ids = []
    for run in runs:
        run_id = train_and_log_model(data_path=data_path, params=run.data.params)
        run_ids.append(run_id)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.rmse ASC"]
    )[0]

    # Register the best model
    best_run_id = best_run.info.run_id
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="random-forest-pricing-lowest-rmse")

if __name__ == '__main__':
    run_register_model()