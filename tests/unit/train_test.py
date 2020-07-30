from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tests.unit.fixtures import cv_results, data
from tests.unit.mocks import MockDataset, MockRunContext, MockWorkspace

from src.train import load_data, main, preprocess_data, train_model


@patch("src.train.run", MockRunContext())
@patch("src.train.Workspace", MockWorkspace())
@patch("src.train.Dataset", MockDataset())
def test_load_data():
    # Define target dataframe and returned dataframe after loading data
    target_df = pd.DataFrame(data.copy())
    return_df = load_data(None)

    # Should return desired number of columns
    assert len(return_df.columns) == len(target_df.columns)

    # Should contain all desired columns
    assert set(return_df.columns) == set(target_df.columns)


@patch("src.train.run", MockRunContext())
def test_preprocess_data():
    # Return dataframe after processing data
    input_df = pd.DataFrame(data.copy())
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.run", MockRunContext())
def test_preprocess_data_nulls():
    # Create dataset with additional record with null
    data_with_null = data.copy()
    null_record = data_with_null[0]
    null_record["age"] = np.nan
    data_with_null.append(null_record)

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_null)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.run", MockRunContext())
def test_preprocess_data_duplicates():
    # Create dataset with additional duplicate record
    data_with_dup = data.copy()
    data_with_dup.append(data_with_dup[0])

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_dup)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.run", MockRunContext())
@patch("src.train.cross_validate")
def test_train_model(mock_cross_validate):
    # Mock retuirn value of cross_validate
    mock_cross_validate.return_value = cv_results

    # Train model
    input_df = pd.DataFrame(data.copy())
    df = preprocess_data(input_df)
    model = train_model(df)

    # Should return an sklearn pipeline
    assert type(model) == Pipeline


@patch("src.train.run", MockRunContext())
@patch("src.train.parse_args")
@patch("src.train.load_data")
@patch("src.train.cross_validate")
@patch("src.train.os.makedirs", MagicMock())
@patch("src.train.joblib.dump")
def test_main(mock_dump, mock_cross_validate, mock_load_data, mock_parse_args):
    # Mock retuirn values
    mock_parse_args.return_value = "dataset_name"
    mock_cross_validate.return_value = cv_results
    mock_load_data.return_value = pd.DataFrame(data.copy())

    # Execute main
    main()

    # Should have made a call to write model
    mock_dump.assert_called_once()
