from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.score import process_data, run
from tests.fixtures.data import data_int
from tests.mocks.model import MockModel


def test_process_data():
    # generate payload
    payload = data_int[0]
    payload.pop('cardiovascular_disease', None)

    # apply preprocessing
    input_df = pd.DataFrame([payload])
    X = process_data(input_df)

    # calculate BMI value
    payload_bmi = payload['weight'] / (payload['height'] / 100) ** 2

    # should return a dataframe with 1 row and 10 columns
    assert X.shape == (1, 10)

    # should include column for BMI
    assert 'bmi' in X.columns.tolist()

    # should remove height and weight columns
    assert 'height' not in X.columns.tolist()
    assert 'weight' not in X.columns.tolist()

    # should contain correct BMI value
    assert X.iloc[0].bmi == payload_bmi


@patch('scripts.score.model', MockModel())
def test_run():
    # generate payload
    payload = data_int[0]
    payload.pop('cardiovascular_disease', None)

    # return prediction
    result = run([payload])
    prediction_probabilities = [[0.7581071779416333, 0.24189282205836665]]

    # should return a dictionary
    assert type(result) == dict

    # should return dictionary with key of 'predict_proba' and value equal to a list of '<probabilities>'
    assert 'predict_proba' in result.keys()
    assert type(result['predict_proba']) == list
    assert result['predict_proba'] == prediction_probabilities
