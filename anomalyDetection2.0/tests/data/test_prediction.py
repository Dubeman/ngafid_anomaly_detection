import os
import numpy as np
import pytest


#any method that starts with test_ is a test method and will be run by pytest



class TestPrediction:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.normal_features_path = "/Users/manasdubey2022/Desktop/NGAFID/Codebase/anomalyDetection2.0/extracted_features/normal_features_100.npy"
        self.anomalous_features_path = "/Users/manasdubey2022/Desktop/NGAFID/Codebase/anomalyDetection2.0/extracted_features/anomalous_features_100.npy"
        self.normal_features = None
        self.anomalous_features = None

    def load_features(self, features_path, num_kernels, normal):
        """Load features from file."""
        features = np.load(features_path)
        if normal:
            assert features.shape[0] == 976, "Wrong number of normal features"
        else:
            assert features.shape[0] == 1413, "Wrong number of anomalous features"

        assert features.shape[1] == num_kernels * 2, "Wrong feature dimension"
        return features

    def test_load_normal_features(self):
        self.normal_features = self.load_features(self.normal_features_path, num_kernels=100, normal=True)

    def test_load_anomalous_features(self):
        self.anomalous_features = self.load_features(self.anomalous_features_path, num_kernels=100, normal=False)





if __name__ == "__main__":
    pytest.main([__file__])



        