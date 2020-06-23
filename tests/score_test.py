from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.score import process_data, run
from tests.fixtures import data_int
from tests.mocks import MockModel

print(os.path.dirname(os.path.abspath(__file__)))


def test_process_data():
    # Generate payload
    payload = data_int[0]
    payload.pop('cardiovascular_disease', None)

    # Apply preprocessing
    input_df = pd.DataFrame([payload])
    X = process_data(input_df)

    # Calculate BMI value
    payload_bmi = payload['weight'] / (payload['height'] / 100) ** 2

    # Should return a dataframe with 1 row and 10 columns
    assert X.shape == (1, 10)

    # Should include column for BMI
    assert 'bmi' in X.columns.tolist()

    # Should remove height and weight columns
    assert 'height' not in X.columns.tolist()
    assert 'weight' not in X.columns.tolist()

    # Should contain correct BMI value
    assert X.iloc[0].bmi == payload_bmi


@patch('scripts.score.model', MockModel())
def test_run():
    # Generate payload
    payload = data_int[0]
    payload.pop('cardiovascular_disease', None)

    # Return prediction
    result = run([payload])
    prediction_probabilities = [[0.7581071779416333, 0.24189282205836665]]

    # Should return a dictionary
    assert type(result) == dict

    # Should return dictionary with key of 'predict_proba' and value equal to a list of '<probabilities>'
    assert 'predict_proba' in result.keys()
    assert type(result['predict_proba']) == list
    assert result['predict_proba'] == prediction_probabilities
