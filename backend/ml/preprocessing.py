import numpy as np

class FeatureScaler:
    """
    StandardScaler style normalization for input feature vectors.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, vector: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            # Fallback simple z-score per sample if scaler statistics not fitted
            std_val = np.std(vector)
            if std_val < 1e-6:
                return vector - np.mean(vector)
            return (vector - np.mean(vector)) / std_val
        
        return (vector - self.mean) / (self.std + 1e-7)

feature_scaler = FeatureScaler()
