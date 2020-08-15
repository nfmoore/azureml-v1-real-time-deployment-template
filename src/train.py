import os
import sys
import traceback
from argparse import ArgumentParser

import joblib
import numpy as np
import sklearn
from azureml.core import Run
from azureml.core.model import Model
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

run = None
evaluation_metric_threshold = 0.7


def load_data():
    # Retreive dataset
    dataset = run.input_datasets["InputDataset"]

    # Convert dataset to pandas dataframe
    df = dataset.to_pandas_dataframe()

    # Convert strings to float
    df = df.astype(
        {
            "age": np.float64,
            "height": np.float64,
            "weight": np.float64,
            "systolic": np.float64,
            "diastolic": np.float64,
            "cardiovascular_disease": np.float64,
        }
    )

    return df


def preprocess_data(df):
    # Remove missing values
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove records where height or weight is more than 6 std from mean
    df = df[(np.abs(stats.zscore(df.height)) < 6)]
    df = df[(np.abs(stats.zscore(df.weight)) < 6)]

    # Create feature for Body Mass Index (indicator of heart health)
    df["bmi"] = df.weight / (df.height / 100) ** 2

    return df


def train_model(df):
    # Define categorical / numeric features
    categorical_features = [
        "gender",
        "cholesterol",
        "glucose",
        "smoker",
        "alcoholic",
        "active",
    ]
    numeric_features = ["age", "systolic", "diastolic", "bmi"]

    # Get model features / target
    X = df.drop(labels=["height", "weight", "cardiovascular_disease"], axis=1)
    y = df.cardiovascular_disease

    # Convert data types of model features
    X[categorical_features] = X[categorical_features].astype(np.object)
    X[numeric_features] = X[numeric_features].astype(np.float64)

    # Define model pipeline
    scaler = StandardScaler()
    onehotencoder = OneHotEncoder(categories="auto")
    classifier = LogisticRegression(random_state=0, solver="liblinear")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", scaler, numeric_features),
            ("categorical", onehotencoder, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Train / evaluate performance of logistic regression classifier
    cv_results = cross_validate(pipeline, X, y, cv=10, return_train_score=True)

    # Get average train / test accuracy
    train_accuracy = round(cv_results["train_score"].mean(), 4)
    test_accuracy = round(cv_results["test_score"].mean(), 4)

    # Log average train / test accuracy
    run.log("train_acccuracy", train_accuracy)
    run.log("test_acccuracy", test_accuracy)

    # Log performance metrics for data
    for metric in cv_results.keys():
        run.log_row(
            "k_fold_cv_metrics",
            metric=metric.replace("_", " "),
            mean="{:.2}".format(cv_results[metric].mean()),
            std="{:.2}".format(cv_results[metric].std()),
        )

    # Fit model
    pipeline.fit(X, y)

    return pipeline, test_accuracy


def register_model(model_name, build_id, test_acccuracy, model_path):
    # Retreive train datasets
    train_dataset = run.input_datasets["InputDataset"]

    # Define model tags
    model_tags = {
        "build_id": build_id,
        "test_acccuracy": test_acccuracy,
    }

    print("Variable [model_tags]:", model_tags)

    # Register the model
    model = run.register_model(
        model_name=model_name,
        model_path=model_path,
        model_framework=Model.Framework.SCIKITLEARN,
        model_framework_version=sklearn.__version__,
        datasets=train_dataset,
        tags=model_tags,
    )

    print("Variable [model]:", model.serialize())


def parse_args(argv):
    ap = ArgumentParser("train")
    ap.add_argument("--BUILD_ID", dest="build_id", required=True)
    ap.add_argument("--MODEL_NAME", dest="model_name", required=True)

    args, _ = ap.parse_known_args(argv)
    return args


def main():
    try:
        global run

        # Retrieve current service context
        run = Run.get_context()

        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Print argument values
        print("Argument [build_id]:", args.build_id)
        print("Argument [model_name]:", args.model_name)

        # Load data, pre-process data, train and evaluate model
        df = load_data()
        df = preprocess_data(df)
        model, test_accuracy = train_model(df)

        # Define model file name
        model_file_name = "model.pkl"
        output_path = os.path.join("outputs", model_file_name)

        # Upload model file to run outputs for history
        os.makedirs("outputs", exist_ok=True)
        joblib.dump(value=model, filename=output_path)

        # Upload model to run
        run.upload_file(name=model_file_name, path_or_stream=output_path)

        # Register model if performance is better than threshold or cancel run
        if test_accuracy > evaluation_metric_threshold:
            register_model(
                args.model_name, args.build_id, test_accuracy, model_file_name
            )
        else:
            run.cancel()

        run.complete()

    except Exception:
        exception = f"Exception: train.py\n{traceback.format_exc()}"
        print(exception)
        exit(1)


if __name__ == "__main__":
    main()
