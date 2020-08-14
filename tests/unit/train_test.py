from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.train import load_data, main, preprocess_data, train_model


@patch("azureml.data.TabularDataset")
@patch("src.train.run")
def test_load_data(mock_run, mock_tabular_dataset, input_df):
    # Mock tabular dataset return value
    mock_tabular_dataset.to_pandas_dataframe.return_value = input_df

    # Mock run return values
    mock_run._run_id = "run_id_value"
    mock_run.input_datasets = {"InputDataset": mock_tabular_dataset}

    # Define returned dataframe after loading data
    return_df = load_data()

    # Should return desired number of columns
    assert len(return_df.columns) == len(input_df.columns)

    # Should contain all desired columns
    assert set(return_df.columns) == set(input_df.columns)


@patch("src.train.run", MagicMock())
def test_preprocess_data(input_df):
    # Return dataframe after processing data
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.run", MagicMock())
def test_preprocess_data_nulls(data):
    # Create dataset with additional record with null
    null_record = data[0]
    null_record["age"] = np.nan
    data.append(null_record)

    # Return dataframe after processing data
    input_df = pd.DataFrame(data)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.run", MagicMock())
def test_preprocess_data_duplicates(data):
    # Create dataset with additional duplicate record
    data.append(data[0])

    # Return dataframe after processing data
    input_df = pd.DataFrame(data)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert "bmi" in df.columns.tolist()


@patch("src.train.cross_validate")
@patch("src.train.run", MagicMock())
def test_train_model(mock_cross_validate, input_df, cv_results):
    # Mock retuirn value of cross_validate
    mock_cross_validate.return_value = cv_results

    # Train model
    df = preprocess_data(input_df)
    model = train_model(df)

    # Should return an sklearn pipeline
    assert type(model) == Pipeline


@patch("src.train.Run", MagicMock())
@patch("src.train.os.makedirs", MagicMock())
@patch("src.train.load_data")
@patch("src.train.cross_validate")
@patch("src.train.joblib.dump")
def test_main(mock_dump, mock_cross_validate, mock_load_data, input_df, cv_results):
    # Mock retuirn values
    mock_cross_validate.return_value = cv_results
    mock_load_data.return_value = input_df

    # Execute main
    main()

    # Should have made a call to write model
    mock_dump.assert_called_once()
