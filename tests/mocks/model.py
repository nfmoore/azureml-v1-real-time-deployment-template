import numpy as np


class MockModel:

    @staticmethod
    def predict_proba(X):
        return np.array([[0.7581071779416333, 0.24189282205836665]])
