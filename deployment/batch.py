#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

models_folder = Path('output')
models_folder.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")

# Read run_id from file
with open("../orchestration/run_id.txt", "r") as f:
    run_id = f.read().strip()

def read_dataframe(filename: str):
    df = pd.read_csv(filename)

    return df

def prepare_dicts(df: pd.DataFrame):
    categorical = ['ocean_proximity']
    numerical = ['median_income']
    dicts = df[categorical + numerical].to_dict(orient='records')

    return dicts

def load_model(run_id):
    logged_model = f"runs:/{run_id}/models_mlflow"
    model = mlflow.sklearn.load_model(logged_model)

    return model

def apply_model(input_file, run_id, output_file):
    df = read_dataframe(input_file)
    dicts = prepare_dicts(df)

    # Load DictVectorizer
    with open("../orchestration/models/preprocessor.b", "rb") as f_in:
        dv = pickle.load(f_in)
    X = dv.transform(dicts)

    model = load_model(run_id)
    y_pred = model.predict(X)

    df_result = pd.DataFrame()
    df_result['housing_median_age'] = df['housing_median_age']
    df_result['median_income'] = df['median_income']
    df_result['real_median_house_value'] = df['median_house_value']
    df_result['predicted_value'] = y_pred
    df_result['diff'] = df_result['real_median_house_value'] - df_result['predicted_value']
    df_result['model_version'] = run_id

    df_result.to_csv(output_file, index=False)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", default="housing.csv", help="Input CSV file name")
    parser.add_argument("--output_name", default="housing_output.csv", help="Output CSV file name")
    args = parser.parse_args()

    input_file = f'../orchestration/datasets/housing/{args.input_name}'
    output_file = f'output/{args.output_name}'

    apply_model(input_file=input_file, run_id=run_id, output_file=output_file)

if __name__ == "__main__":
    try:
        run()
        print("Batch processing completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import sys
        sys.exit(1)