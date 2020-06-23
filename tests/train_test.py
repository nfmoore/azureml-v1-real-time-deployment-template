from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.train import load_data, preprocess_data, train_model
from tests.fixtures import data_int, data_raw
from tests.mocks import (MockDataset, MockPandasDataFrameDataset,
                         MockRunContext, MockWorkspace)

print(os.path.dirname(os.path.abspath(__file__)))


@patch('scripts.train.Workspace', MockWorkspace())
@patch('scripts.train.Dataset', MockDataset())
def test_load_data():
    # Mock run context
    run = MockRunContext()

    # Define target dataframe and returned dataframe after loading data
    target_df = pd.DataFrame(data_int)
    return_df = load_data(None, run)

    # Should return desired number of columns
    assert len(return_df.columns) == len(target_df.columns)

    # Should contain all desired renamed columns
    assert set(return_df.columns) == set(target_df.columns)

    # Should change values of 'active' column
    assert set(return_df.active.unique().tolist()) <= set(
        target_df.active.unique().tolist())

    # Should change values of 'alcoholic' column
    assert set(return_df.alcoholic.unique().tolist()) <= set(
        target_df.alcoholic.unique().tolist())

    # Should change values of 'cholesterol' column
    assert set(return_df.cholesterol.unique().tolist()) <= set(
        target_df.cholesterol.unique().tolist())

    # Should change values of 'gender' column
    assert set(return_df.gender.unique().tolist()) <= set(
        target_df.gender.unique().tolist())

    # Should change values of 'glucose' column
    assert set(return_df.glucose.unique().tolist()) <= set(
        target_df.glucose.unique().tolist())

    # Should change values of 'smoker' column
    assert set(return_df.smoker.unique().tolist()) <= set(
        target_df.smoker.unique().tolist())


def test_preprocess_data():
    # Mock run context
    run = MockRunContext()

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_int.copy())
    df = preprocess_data(input_df, run)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


def test_preprocess_data_nulls():
    # Mock run context
    run = MockRunContext()

    # Create dataset with additional record with null
    data_with_null = data_int.copy()
    null_record = data_with_null[0]
    null_record['age'] = np.nan
    data_with_null.append(null_record)

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_null)
    df = preprocess_data(input_df, run)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


def test_preprocess_data_dups():
    # Mock run context
    run = MockRunContext()

    # Create dataset with additional duplicate record
    data_with_dup = data_int.copy()
    data_with_dup.append(data_with_dup[0])

    # Return dataframe after processing data
    input_df = pd.DataFrame(data_with_dup)
    df = preprocess_data(input_df, run)

    # Should return a dataframe with the same rows and an additional columns
    assert df.shape == (input_df.shape[0], input_df.shape[1] + 1)

    # Should include column for BMI
    assert 'bmi' in df.columns.tolist()


def test_train_model():
    # Mock run context
    run = MockRunContext()

    # Train model
    input_df = pd.DataFrame(data_int.copy())
    df = preprocess_data(input_df, run)
    model = train_model(df, run, return_results=False)

    # Should return an sklearn pipeline
    assert type(model) == Pipeline
