import json
import os

import joblib
import numpy as np
import pandas as pd
from azureml.monitoring import ModelDataCollector
from inference_schema.parameter_types.standard_py_parameter_type import \
    StandardPythonParameterType
from inference_schema.schema_decorators import input_schema, output_schema

input_sample = [{'age': 50, 'gender': 'female', 'systolic': 110,
                 'diastolic': 80, 'height': 175, 'weight': 80,
                 'cholesterol': 'normal', 'glucose': 'normal',
                 'smoker': 'not-smoker', 'alcoholic': 'not-alcoholic',
                 'active': 'active'}]
output_sample = {'probability': [0.26883566156891225]}


def init():
    global model
    global inputs_dc, prediction_dc

    # Retreive path to model folder
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')

    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

    # Initialize data collectors
    inputs_dc = ModelDataCollector(
        model_name="cardiovascular_disease_model",
        feature_names=input_sample[0].keys())
    prediction_dc = ModelDataCollector(
        model_name="cardiovascular_disease_model",
        feature_names=output_sample[0].keys())


def process_data(input_df):
    # Define categorical / numeric features
    categorical_features = ['gender', 'cholesterol',
                            'glucose', 'smoker', 'alcoholic', 'active']
    numeric_features = ['age', 'systolic', 'diastolic', 'bmi']

    # Create feature for Body Mass Index (indicator of heart health)
    input_df['bmi'] = input_df.weight / (input_df.height / 100) ** 2

    # Get model features / target
    X = input_df.drop(labels=['height', 'weight'], axis=1)

    # Convert data types of model features
    X[categorical_features] = X[categorical_features].astype(np.object)
    X[numeric_features] = X[numeric_features].astype(np.float64)

    return X


@input_schema('data', StandardPythonParameterType(input_sample))
@output_schema(StandardPythonParameterType(output_sample))
def run(data):
    try:
        # Preprocess payload and get model prediction
        input_df = pd.DataFrame(data)
        X = process_data(input_df)
        proba = model.predict_proba(X)
        result = proba[:, 1].tolist()

        # Log input and prediction to appinsights
        print(json.dumps({'input': data, 'probability': result}))

        # Collect features and prediction data
        inputs_dc.collect(data)
        prediction_dc.collect(result)

        return {'probability': result}

    except Exception as e:
        # Log exception to appinsights
        print(json.dumps({'error': str(e)}))

        # Retern exception
        return {'error': "Internal server error"}
