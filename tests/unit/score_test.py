from unittest.mock import MagicMock, patch

import pandas as pd
from tests.unit.fixtures import data
from tests.unit.mocks import MockModel

from src.score import process_data, run


def test_process_data():
    # Generate payload
    payload = data[0]
    payload.pop("cardiovascular_disease", None)

    # Apply preprocessing
    input_df = pd.DataFrame([payload])
    X = process_data(input_df)

    # Calculate BMI value
    payload_bmi = payload["weight"] / (payload["height"] / 100) ** 2

    # Should return a dataframe with 1 row and 10 columns
    assert X.shape == (1, 10)

    # Should include column for BMI
    assert "bmi" in X.columns.tolist()

    # Should remove height and weight columns
    assert "height" not in X.columns.tolist()
    assert "weight" not in X.columns.tolist()

    # Should contain correct BMI value
    assert X.iloc[0].bmi == payload_bmi


@patch("src.score.inputs_dc", MagicMock())
@patch("src.score.prediction_dc", MagicMock())
@patch("src.score.model", MockModel())
def test_run():
    # Generate payload
    payload = data[0]
    payload.pop("cardiovascular_disease", None)

    # Return prediction
    result = run([payload])
    prediction_probabilities = [0.24189282205836665]

    # Should return a dictionary
    assert type(result) == dict

    # Should return valid response payload
    assert "probability" in result.keys()
    assert type(result["probability"]) == list
    assert result["probability"] == prediction_probabilities
