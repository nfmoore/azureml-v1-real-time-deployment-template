import os
from argparse import ArgumentParser

import joblib
import numpy as np
from azureml.core import Dataset, Run, Workspace
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

run = None


def load_data(dataset_name):
    # Retreive dataset
    if run._run_id.startswith('OfflineRun'):
        workspace = Workspace.from_config()
        dataset = Dataset.get_by_name(workspace, name=dataset_name)
    else:
        workspace = run.experiment.workspace
        dataset = Dataset.get_by_name(workspace, name=dataset_name)

    # Convert dataset to pandas dataframe
    df = dataset.to_pandas_dataframe()

    # Convert strings to float
    df = df.astype({
        'age': np.float64, 'height': np.float64, 'weight': np.float64,
        'systolic': np.float64, 'diastolic': np.float64,
        'cardiovascular_disease': np.float64})

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
    df['bmi'] = df.weight / (df.height / 100) ** 2

    return df


def train_model(df):
    # Define categorical / numeric features
    categorical_features = ['gender', 'cholesterol',
                            'glucose', 'smoker', 'alcoholic', 'active']
    numeric_features = ['age', 'systolic', 'diastolic', 'bmi']

    # Get model features / target
    X = df.drop(labels=['height', 'weight', 'cardiovascular_disease'], axis=1)
    y = df.cardiovascular_disease

    # Convert data types of model features
    X[categorical_features] = X[categorical_features].astype(np.object)
    X[numeric_features] = X[numeric_features].astype(np.float64)

    # Define model pipeline
    scaler = StandardScaler()
    onehotencoder = OneHotEncoder(categories='auto')
    classifier = LogisticRegression(random_state=0, solver="liblinear")

    preprocessor = ColumnTransformer(transformers=[
        ('numeric', scaler, numeric_features),
        ('categorical', onehotencoder, categorical_features)
    ], remainder="drop")

    pipeline = Pipeline(
        steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    # Train / evaluate performance of logistic regression classifier
    cv_results = cross_validate(
        pipeline, X, y, cv=10, return_train_score=True)

    # Log average train / test accuracy
    run.log('Average Train Acccuracy',
            round(cv_results['train_score'].mean(), 4))
    run.log('Average Test Acccuracy',
            round(cv_results['test_score'].mean(), 4))

    # Log performance metrics for data
    for metric in cv_results.keys():
        run.log_row(
            "K-Fold CV Metrics",
            metric=metric.replace('_', ' '),
            mean='{:.2%}'.format(cv_results[metric].mean()),
            std='{:.2%}'.format(cv_results[metric].std()))

    # Fit model
    pipeline.fit(X, y)

    return pipeline


def parse_args():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--DATASET_NAME', required=True)
    args = parser.parse_args()

    return args.DATASET_NAME


def main():
    try:
        # Retrieve current service context
        global run
        run = Run.get_context()

        # Retrieve model name and dataset name from runtime arguments
        dataset_name = parse_args()

        # Load data, pre-process data, train and evaluate model
        df = load_data(dataset_name)
        df = preprocess_data(df)
        model = train_model(df)

        # Save the model to the outputs directory for capture
        os.makedirs('./outputs', exist_ok=True)
        joblib.dump(value=model, filename='./outputs/model.pkl')

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
