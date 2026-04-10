from sklearn.ensemble import RandomForestClassifier
import numpy as np

class AudioModel:
    def __init__(self):
        self.model = RandomForestClassifier()

    def preprocess_features(self, mfcc):
        # Convert (13, time) → (13,)
        return np.mean(mfcc, axis=1)

    def train(self, X, y):
        X_processed = [self.preprocess_features(x) for x in X]
        self.model.fit(X_processed, y)

    def predict(self, mfcc):
        features = self.preprocess_features(mfcc)
        return self.model.predict([features])[0]