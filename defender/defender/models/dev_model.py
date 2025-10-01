import joblib
import pefile
import numpy as np
import envparse
from defender.models.pe_extraction_dev_v2 import PEFeatureExtractorLief

# extractor = PEFeatureExtractorLief(section_entropy_bins=10)
extractor = PEFeatureExtractorLief()

THRESHOLD_ENV = envparse.env("THRESHOLD_ENV", cast=float, default=0.5)
DEV_MODEL_PATH = "defender/models/dev_model.pkl"

class DevModel(object):
    def __init__(self):
        saved_dev_model = joblib.load(DEV_MODEL_PATH)
        self.model = saved_dev_model


    def predict(self, bytez: bytes) -> int:
        features, names = extractor.extract(bytez)

        X = np.array(features).reshape(1, -1)  # 1 sample, n_features columns
        probs = self.model.predict_proba(X)[:, 1]  # probability of malware
        y_pred = (probs > THRESHOLD_ENV).astype(int)
        res = int(self.model.predict(X)[0])
        print(f"Threshold Result: {y_pred}\tNon-Threshold Result: {res}\tScore: {probs}")
        return int(y_pred)

    def model_info(self):
        return {"thresh": self.thresh,
                "name": self.__name__}