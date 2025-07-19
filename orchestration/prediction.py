#!/usr/bin/env python
# coding: utf-8

import os
import tarfile
import urllib
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("california-housing-experiment")

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = 'datasets/housing/'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    '''Gets the data for the model'''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    '''Loads the data'''
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

def split_train_test(data, test_ratio):
    '''Separates the data for train and test purpose'''
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def create_X(df, dv=None):
    '''Creates the feature matrix X from the dataframe df'''
    categorical = ['ocean_proximity']
    numerical = ['median_income']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        best_params = {
            'max_depth': 7,
            'min_samples_leaf': 3,
            'min_samples_split': 9,
            'n_estimators': 13,
            'random_state': 42
        }

        mlflow.log_params(best_params)

        new_params = {}
        for param in best_params:
            new_params[param] = int(best_params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        mlflow.log_metric("val_rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(rf, artifact_path="models_mlflow")

        return run.info.run_id

def run():
    fetch_housing_data()
    housing = load_housing_data()

    train_set, test_set = split_train_test(housing, 0.2)

    X_train, dv = create_X(train_set)
    X_val, _ = create_X(test_set, dv)
    
    target = 'median_house_value'
    y_train = train_set[target].values
    y_val = test_set[target].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    
    return run_id

if __name__ == "__main__":
    run_id = run()

    with open("run_id.txt", "w") as f:
        f.write(run_id)