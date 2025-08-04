import os
import pickle
import mlflow.sklearn

def test_model_file_exists():
    # Check if the model file exists
    model_path = "main/models/preprocessor.b"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

def test_model_predict_shape():
    mlflow.set_tracking_uri("http://localhost:5000")
    # Load the model and DictVectorizer
    with open("main/models/preprocessor.b", "rb") as f:
        dv = pickle.load(f)
    # Create a dummy input
    sample = [{"ocean_proximity": "INLAND", "median_income": 3.0}]
    X = dv.transform(sample)
    # Load the trained model
    with open("orchestration/run_id.txt") as f:
        run_id = f.read().strip()
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/models_mlflow")
    # Predict
    y_pred = model.predict(X)
    assert y_pred.shape[0] == X.shape[0], "Prediction output shape does not match input"