import os
import pandas as pd
import tqdm
from sklearn import svm 
# from dev_pe_extraction import extract_features
import joblib
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV


from dev_pe_extraction_v2 import PEFeatureExtractorLief
# extractor = PEFeatureExtractorLief(section_entropy_bins=10)
extractor = PEFeatureExtractorLief()


SAMPLES_PATH = "samples-training/pe-machine-learning-dataset/pe-machine-learning-dataset/samples"
SAMPLES_CSV_PATH = "samples-training/pe-machine-learning-dataset/pe-machine-learning-dataset/samples.csv"

df = pd.read_csv(SAMPLES_CSV_PATH)

X = []
Y = []
count =0 
for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    count += 1
    if count % 100 != 0:
        continue
    file_id = str(row["id"])  # filenames in SAMPLES_PATH
    label = row["list"]       # "Whitelist" or "Blacklist"

    fpath = os.path.join(SAMPLES_PATH, file_id)

    if not os.path.isfile(fpath):
        print(f"ERROR: {fpath} not found.")
        continue  # skip missing files

    # feats = extract_features(fpath)
    p = Path(fpath)
    data = p.read_bytes()
    feats, names = extractor.extract(data)
    if feats is not None and feats.size > 0:
        X.append(feats)
        if label == "Whitelist":
            Y.append(0)  # goodware
        elif label == "Blacklist":
            Y.append(1)  # malware

# Instantiate a classifier and train it with the vectors
clf = SGDClassifier(
    loss="hinge",      # linear SVM
    max_iter=1000,     # number of epochs
    tol=1e-3,          # stopping criterion
    n_jobs=-1,         # use all CPU cores
    random_state=42
)
calibrated_clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
calibrated_clf.fit(X, Y)


joblib.dump(calibrated_clf, "../defender/defender/models/dev_model.pkl") 





