import argparse
import os

import joblib
import numpy as np
from azureml.core import Dataset, Run, Workspace
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(dataset_name, run):
    # Retreive dataset
    if run._run_id.startswith('_OfflineRun'):
        run = None

    if run is None:
        workspace = Workspace.from_config()
        # dataset = Dataset.get_by_name(workspace, name=dataset_name)
    else:
        workspace = run.experiment.workspace

    # Convert dataset to pandas dataframe
    dataset = Dataset.get_by_name(workspace, name=dataset_name)
    df = dataset.to_pandas_dataframe()

    # Rename features
    df.rename(
        columns={'ap_hi': 'systolic', 'ap_lo': 'diastolic', 'gluc': 'glucose',
                 'smoke': 'smoker', 'alco': 'alcoholic',
                 'cardio': 'cardiovascular_disease'}, inplace=True)

    # Convert age to years
    df.age = df.age.apply(lambda x: x / 365)

    # Convert categorical data values
    df.gender.replace({1: 'male', 2: 'female'}, inplace=True)
    df.cholesterol.replace({1: 'normal', 2: 'above-normal', 3:
                            'well-above-normal'}, inplace=True)
    df.glucose.replace({1: 'normal', 2: 'above-normal',
                        3: 'well-above-normal'},  inplace=True)
    df.gender.replace({1: 'male', 2: 'female'}, inplace=True)
    df.smoker.replace({0: 'non-smoker', 1: 'smoker'}, inplace=True)
    df.alcoholic.replace({0: 'not-alcoholic', 1: 'alcoholic'}, inplace=True)
    df.active.replace({0: 'not-active', 1: 'active'}, inplace=True)

    return df


def preprocess_data(df, run):

    # Remove missing values
    df.dropna(inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Remove records where height or weight is more than 6 std from mean
    df = df[(np.abs(stats.zscore(df.height)) < 6)]
    df = df[(np.abs(stats.zscore(df.weight)) < 6)]

    # Create feature for Body Mass Index (indicator of heart health)
    df['bmi'] = df.weight / (df.height / 100) ** 2

    # Log summary statistics for data
    # run.log_table("Data Description", df.describe().to_dict())

    return df


def train_model(df, run, return_results=True):
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

    if return_results:
        pass
        # Train / evaluate performance of logistic regression classifier
        # cv_results = cross_validate(
        #     pipeline, X, y, cv=10, return_train_score=True)

        # Log performance metrics for data
        # for metric in cv_results.keys():
        #     run.log_row(
        #         "Performance Metrics", metric=metric.replace('_', ' '),
        #         mean=cv_results[metric].mean(), std=cv_results[metric].std())

        # Fit model

    pipeline.fit(X, y)

    return pipeline


def getRuntimeArgs():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_NAME',  default='cvd-model')
    parser.add_argument('--DATASET_NAME', default='cardiovascular-disease')
    args = parser.parse_args()

    return args.MODEL_NAME, args.DATASET_NAME


def main():
    # Retrieve current service context for logging metrics and uploading files
    run = Run.get_context(allow_offline=True)

    # Retrieve model name and dataset name from runtime arguments
    model_name, dataset_name = getRuntimeArgs()

    # Load data, pre-process data, train and evaluate model
    df = load_data(dataset_name, run)
    df = preprocess_data(df, run)
    model = train_model(df, run)

    # Save the model to the outputs directory for capture
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/{}'.format(model_name))


if __name__ == "__main__":
    main()
