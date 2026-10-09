import os
import yaml
import mlflow
import pandas as pd
import pickle
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] ="https://dagshub.com/sheshpalsingh2024/ML_pipeline_end_to_end.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']='sheshpalsingh2024'
os.environ['MLFLOW_TRACKING_PASSWORD'] ='0774b1027b40a4091072d1e585b95c6cddf3d382'


def hyperparameter_tunnig(X_train, y_train, param_grid):
    rf = RandomForestClassifier()

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    return grid_search


# Load the parameter from param.yaml
# Load parameters from params.yaml
with open("../params.yaml", "r") as file:
    params = yaml.safe_load(file)["train"]


def train(data_path, model_path, random_state, n_estimators, max_depth):

    data = pd.read_csv(data_path)

    df = data.copy()

    # Remove rows where the target column is missing
    df = df.dropna(subset=["Outcome"])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    mlflow.set_tracking_uri(
        "https://dagshub.com/sheshpalsingh2024/"
        "ML_pipeline_end_to_end.mlflow"
    )

    # Start MLflow run
    with mlflow.start_run():

        # Split data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=0.8,
            random_state=random_state
        )

        signature = infer_signature(X_train, y_train)

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tunnig(
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid
        )

        # Get the best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate
        y_pred = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.4f}")

        # Log additional metrics
        mlflow.log_metric("Accuracy", accuracy)

        mlflow.log_params(grid_search.best_params_)

        # Log the confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)

        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")

        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(
            mlflow.get_tracking_uri()
        ).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                signature=signature,
                registered_model_name="best_model"
            )
        else:
            mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature
        )

        # Create the directory to save the model
        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        # Save the trained model
        with open(model_path, "wb") as file:
            pickle.dump(best_model, file)

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train(
        params["data"],
        params["model"],
        params["random_state"],
        params["n_estimators"],
        params["max_depth"]
    )