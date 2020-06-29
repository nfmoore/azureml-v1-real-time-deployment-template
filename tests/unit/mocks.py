import numpy as np
import pandas as pd

from tests.unit.fixtures import data_raw


class MockModel:

    @staticmethod
    def predict_proba(X):
        return np.array([[0.7581071779416333, 0.24189282205836665]])


class MockPandasDataFrameDataset:

    @staticmethod
    def to_pandas_dataframe():
        return pd.DataFrame(data_raw)


class MockRunContext:
    _run_id = 'OfflineRun'

    @staticmethod
    def log_row(name, description=None, **kwargs):
        pass

    @staticmethod
    def log(name, value, description=''):
        pass


class MockDataset:

    @staticmethod
    def get_by_name(workspace, name):
        return MockPandasDataFrameDataset()


class MockWorkspace:

    @staticmethod
    def from_config():
        return None
