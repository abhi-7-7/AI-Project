import numpy as np
from sklearn.ensemble import RandomForestClassifier


class AudioModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def preprocess_features(self, mfcc):
        # Mean
        mean = np.mean(mfcc, axis=1)

        # Standard deviation
        std = np.std(mfcc, axis=1)

        # Combine
        features = np.concatenate((mean, std))

        return features

    def train(self, X, y):
        X_processed = [self.preprocess_features(x) for x in X]
        self.model.fit(X_processed, y)

    def predict(self, mfcc):
        features = self.preprocess_features(mfcc)
        return self.model.predict([features])[0]