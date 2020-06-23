import numpy as np
import pandas as pd

df_fixture = [{
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


def process_data(input_df):
    categorical_features = ['gender', 'cholesterol',
                            'glucose', 'smoker', 'alcoholic', 'active']
    numeric_features = ['age', 'systolic', 'diastolic', 'bmi']
    input_df['bmi'] = input_df.weight / (input_df.height / 100) ** 2
    X = input_df.drop(labels=['height', 'weight'], axis=1)
    X[categorical_features] = X[categorical_features].astype(np.object)
    X[numeric_features] = X[numeric_features].astype(np.float64)
    return X


def test_process_data():
    df = pd.DataFrame(df_fixture)
    X = process_data(df)

    np.testing.assert_almost_equal(X.shape, (1, 10))
    np.testing.assert_almost_equal('bmi' in X.columns.tolist(), True)
    np.testing.assert_almost_equal('height' not in X.columns.tolist(), True)
    np.testing.assert_almost_equal('weight' not in X.columns.tolist(), True)
    np.testing.assert_almost_equal(
        df_fixture[0]['weight'] / (df_fixture[0]['height'] / 100) ** 2,
        X.iloc[0].bmi)
