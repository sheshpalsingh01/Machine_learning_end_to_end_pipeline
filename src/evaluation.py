import os
import mlflow
import pandas as pd
import pickle
import yaml
from sklearn.metrics import accuracy_score

os.environ['MLFLOW_TRACKING_URI'] ="https://dagshub.com/sheshpalsingh2024/ML_pipeline_end_to_end.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='sheshpalsingh2024'
os.environ['MLFLOW_TRACKING_PASSWORD'] ='0774b1027b40a4091072d1e585b95c6cddf3d382'


# Load parameters from params.yaml
with open("../params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]


def evaluate(data_path, model_path):

    data = pd.read_csv(data_path)

    df = data.copy()

    # Remove rows with missing target values
    df = df.dropna(subset=["Outcome"])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    mlflow.set_tracking_uri(
        "https://dagshub.com/sheshpalsingh2024/"
        "ML_pipeline_end_to_end.mlflow"
    )

    # Load the model from disk
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Generate predictions
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metric("Accuracy", accuracy)

    print(f"Model accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    evaluate(
        params["data"],
        params["model"]
    )