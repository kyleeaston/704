import joblib
import pefile
import numpy as np
DEV_MODEL_PATH = "defender/models/dev_model.pkl"

class DevModel(object):
    def __init__(self, thresh: float = 0.1234, name: str = 'dummy'):
        # with open(DEV_MODEL_PATH, 'rb') as f:
        #     saved_dev_model = pickle.load(f)
        saved_dev_model = joblib.load(DEV_MODEL_PATH)

        self.model = saved_dev_model
        self.thresh = thresh
        self.__name__ = name

    def extract_features(self, data: bytes):
        try:
            pe = pefile.PE(data=data)

            try:
                size_of_code = pe.OPTIONAL_HEADER.SizeOfCode
            except Exception:
                size_of_code = 0

            try:
                num_sections = len(pe.sections)
            except Exception:
                num_sections = 0

            try:
                entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            except Exception:
                entry_point = 0

            try:
                size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
            except Exception:
                size_of_image = 0

            features = np.array(
                [size_of_code, num_sections, entry_point, size_of_image],
                dtype=np.float64
            )

            # Clean up NaN/inf values just in case
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            print(f"Completely failed to parse PE data: {e}")
            # If parsing fails at the top level, return zeros
            return np.zeros(4, dtype=np.float64)

    def predict(self, bytez: bytes) -> int:
        features = self.extract_features(bytez)
        print(features)
        X = np.array(features).reshape(1, -1)  # 1 sample, n_features columns
        print(X)
        res = int(self.model.predict(X)[0])
        return res

    def model_info(self):
        return {"thresh": self.thresh,
                "name": self.__name__}