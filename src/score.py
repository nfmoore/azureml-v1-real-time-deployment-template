import os

import joblib
import numpy as np
import pandas as pd
from inference_schema.parameter_types.standard_py_parameter_type import \
    StandardPythonParameterType
from inference_schema.schema_decorators import input_schema, output_schema

model = None
inputs_dc = None
prediction_dc = None

input_sample = [{'age': 50, 'gender': 'female', 'systolic': 110,
                 'diastolic': 80, 'height': 175, 'weight': 80,
                 'cholesterol': 'normal', 'glucose': 'normal',
                 'smoker': 'not-smoker', 'alcoholic': 'not-alcoholic',
                 'active': 'active'}]
output_sample = {'probability': [0.26883566156891225]}


def init():
    from azureml.monitoring import ModelDataCollector

    global model
    global inputs_dc, prediction_dc

    # Retreive path to model folder
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')

    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

    # Initialize data collectors
    inputs_dc = ModelDataCollector(
        model_name='cardiovascular_disease_model',
        designation='inputs',
        feature_names=['age', 'gender', 'systolic', 'diastolic', 'height',
                       'weight', 'cholesterol', 'glucose', 'smoker',
                       'alcoholic', 'active'])
    prediction_dc = ModelDataCollector(
        model_name="cardiovascular_disease_model",
        designation='predictions',
        feature_names=['cardiovascular_disease'])


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
        print('Request Payload', data)
        print('Response Payload', result)

        # Collect features and prediction data
        inputs_dc.collect(input_df)
        prediction_dc.collect(pd.DataFrame((proba[:, 1] >= 0.5).astype(int),
                                           columns=['cardiovascular_disease']))

        return {'probability': result}

    except Exception as e:
        # Log exception to appinsights
        print('Error', str(e))

        # Retern exception
        return {'error': "Internal server error"}
