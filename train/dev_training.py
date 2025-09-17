import os
import pandas as pd
import tqdm
from sklearn import svm 
from dev_pe_extraction import extract_features
import joblib



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

    feats = extract_features(fpath)
    if feats:
        X.append(feats)
        if label == "Whitelist":
            Y.append(0)  # goodware
        elif label == "Blacklist":
            Y.append(1)  # malware

# Instantiate a classifier and train it with the vectors
clf = svm.SVC() # is this the trained model that will be saved in .pkl????
clf.fit(X, Y)

joblib.dump(clf, "../defender/defender/models/dev_model.pkl") 





