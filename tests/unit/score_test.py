from unittest.mock import MagicMock, patch

import numpy as np

from src.score import process_data, run


def test_process_data(input_df):
    X = process_data(input_df.drop("cardiovascular_disease", axis=1))

    # Calculate BMI value
    payload_bmi = input_df.weight / (input_df.height / 100) ** 2

    # Should return a dataframe with an additional column
    assert X.shape == (input_df.shape[0], input_df.shape[1] - 2)

    # Should include column for BMI
    assert "bmi" in X.columns.tolist()

    # Should remove height and weight columns
    assert "height" not in X.columns.tolist()
    assert "weight" not in X.columns.tolist()

    # Should contain correct BMI value
    assert X.iloc[0].bmi == payload_bmi[0]


@patch("src.score.inputs_dc", MagicMock())
@patch("src.score.prediction_dc", MagicMock())
@patch("src.score.model")
def test_run(mock_model, data):
    # Mock model predictions
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

    # Generate payload
    payload = data[0]
    payload.pop("cardiovascular_disease", None)

    # Return prediction
    result = run([payload])
    prediction_probabilities = [0.3]

    # Should return a dictionary
    assert type(result) == dict

    # Should return valid response payload
    assert "probability" in result.keys()
    assert type(result["probability"]) == list
    assert result["probability"] == prediction_probabilities
