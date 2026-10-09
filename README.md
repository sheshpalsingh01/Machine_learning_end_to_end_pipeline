# ML Pipeline End-to-End

This project implements a simple end-to-end machine learning workflow for a tabular dataset using Python, scikit-learn, DVC, and MLflow. It covers the full lifecycle of a model: data preprocessing, training, evaluation, experiment tracking, and reproducible pipeline execution.

## Overview

The project is organized as a DVC pipeline with three stages:

1. Preprocess raw data
2. Train a Random Forest classifier
3. Evaluate the trained model

The pipeline is parameterized through `params.yaml`, and the trained model is saved in the `models/` directory. Experiment metrics and metadata are tracked with MLflow.

## Project Structure

```text
ML_pipeline_end_to_end/
├── .dvc/                       # DVC metadata
├── data/
│   ├── raw/                   # Raw dataset input
│   └── processed/             # Preprocessed dataset output
├── models/                    # Trained model artifacts
├── src/
│   ├── preprocess.py          # Data cleaning/preprocessing
│   ├── train.py               # Model training and MLflow logging
│   ├── evaluation.py          # Accuracy evaluation and MLflow logging
│   └── mlflow.db              # Local MLflow metadata database
├── .dvcignore
├── .gitingore
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Pipeline parameters
├── requirements.txt           # Python dependencies
├── README.md
└── dvc.lock                   # DVC lock file
```

## Tech Stack

- Python 3.x
- scikit-learn
- pandas
- DVC
- MLflow
- Dagshub (used for remote tracking)

## Dataset

The project expects a CSV dataset containing a target column named `Outcome`. The preprocessing step reads the raw CSV, and training uses the processed data to split features and labels.

## Setup

Create and activate a virtual environment:

```bash
cd ML_pipeline_end_to_end
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, install DVC and MLflow explicitly:

```bash
pip install dvc mlflow dagshub
```

## Configuration

The main parameters are stored in `params.yaml`:

```yaml
preprocess:
  input: ../data/raw/data.csv
  output: ../data/processed/data.csv

train:
  data: ../data/raw/data.csv
  model: ../models/model.pkl
  random_state: 42
  n_estimators: 100
  max_depth: 5
```

These values control pipeline inputs, model output location, and training behavior.

## Running the Pipeline

Run the full DVC pipeline:

```bash
dvc repro
```

View the pipeline graph:

```bash
dvc dag
```

Check pipeline status:

```bash
dvc status
```

## Pipeline Stages

### 1) Preprocess

Script: `src/preprocess.py`

- Reads the raw dataset
- Creates the processed output folder if missing
- Writes the processed CSV to `data/processed/data.csv`

### 2) Train

Script: `src/train.py`

- Loads processed data
- Drops rows with missing target values
- Splits data into train/test sets
- Performs hyperparameter tuning using `GridSearchCV`
- Trains a `RandomForestClassifier`
- Logs metrics and model details to MLflow
- Saves the best model as `models/model.pkl`

### 3) Evaluate

Script: `src/evaluation.py`

- Loads the saved model
- Predicts on test data
- Computes accuracy
- Logs evaluation metrics to MLflow

## Example Commands

Run each stage manually:

```bash
cd src
python preprocess.py
python train.py
python evaluation.py
```

## MLflow Tracking

The project is configured to log experiments to a Dagshub MLflow tracking server. The training and evaluation scripts set the tracking URI and log metrics, parameters, and model artifacts.

If you want to use your own tracking setup, update the environment and URI values in the scripts as needed.

## Notes

- DVC is used to make the ML workflow reproducible and dependency-aware.
- Data files and model artifacts are tracked/managed through the DVC pipeline.
- The repository is best suited for small tabular ML experiments and learning workflows.

## License

This project is intended for educational and learning purposes.
