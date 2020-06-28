from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.train import load_data, main, preprocess_data, train_model
from tests.fixtures import cv_results, data_intermediate
from tests.mocks import MockDataset, MockRunContext, MockWorkspace


@patch('scripts.train.run', MockRunContext())
@patch('scripts.train.Workspace', MockWorkspace())
@patch('scripts.train.Dataset', MockDataset())
def test_load_data():
    # Define target dataframe and returned dataframe after loading data
    target_df = pd.DataFrame(data_intermediate)
    return_df = load_data(None)

    # Should return desired number of columns
    assert len(return_df.columns) == len(target_df.columns)

    # Should contain all desired renamed columns
    assert set(return_df.columns) == set(target_df.columns)

    # Should change values of categorical columns
    for column in target_df.columns:
        assert set(return_df[column].unique().tolist()) <= set(
            target_df[column].unique().tolist())


@patch('scripts.train.run', MockRunContext())
def test_preprocess_data():
    # Return dataframe after processing data
    input_df = pd.DataFrame(data_intermediate.copy())
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


@patch('scripts.train.run', MockRunContext())
def test_preprocess_data_nulls():
    # Create dataset with additional record with null
    data_with_null = data_intermediate.copy()
    null_record = data_with_null[0]
    null_record['age'] = np.nan
    data_with_null.append(null_record)

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_null)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


@patch('scripts.train.run', MockRunContext())
def test_preprocess_data_duplicates():
    # Create dataset with additional duplicate record
    data_with_dup = data_intermediate.copy()
    data_with_dup.append(data_with_dup[0])

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_dup)
    df = preprocess_data(input_df)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


@patch('scripts.train.run', MockRunContext())
@patch('scripts.train.cross_validate')
def test_train_model(mock_cross_validate):
    # Mock retuirn value of cross_validate
    mock_cross_validate.return_value = cv_results

    # Train model
    input_df = pd.DataFrame(data_intermediate.copy())
    df = preprocess_data(input_df)
    model = train_model(df)

    # Should return an sklearn pipeline
    assert type(model) == Pipeline


@patch('scripts.train.run', MockRunContext())
@patch('scripts.train.parse_args')
@patch('scripts.train.load_data')
@patch('scripts.train.cross_validate')
@patch('scripts.train.os.makedirs', MagicMock())
@patch('scripts.train.joblib.dump')
def test_main(mock_dump, mock_cross_validate, mock_load_data, mock_parse_args):
    # Mock retuirn values
    mock_parse_args.return_value = 'dataset_name'
    mock_cross_validate.return_value = cv_results
    mock_load_data.return_value = pd.DataFrame(data_intermediate.copy())

    # Execute main
    main()

    # Should have made a call to write model
    mock_dump.assert_called_once()
