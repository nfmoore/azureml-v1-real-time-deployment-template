import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model
from inference_schema.parameter_types.standard_py_parameter_type import \
    StandardPythonParameterType
from inference_schema.schema_decorators import input_schema, output_schema

model = None

input_sample = [{
    'age': 50.391780821917806,
    'gender': 'female',
    'systolic': 110,
    'diastolic': 80,
    'height': 168,
    'weight': 62.0,
    'cholesterol': 'normal',
    'glucose': 'normal',
    'smoker': 'non-smoker',
    'alcoholic': 'not-alcoholic',
    'active': 'active',
    'bmi': 21.9671201814059
}]
output_sample = {'predict_proba': [[0.7581071779416333, 0.24189282205836665]]}


def init():
    global model

    # Retreive path to model folder
    model_path = Model.get_model_path(
        os.getenv('AZUREML_MODEL_DIR').split('/')[-2])

    # Deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


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

        # Return model prediction in response body
        result = {'predict_proba': proba.tolist()}
        return result
    except Exception as error:
        return {'error': str(error)}
