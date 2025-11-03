import shap
from joblib import load

import matplotlib.pyplot as plt

feature_names = ["Read", "write", "open", "close", "fast read", "fast write", "fast close", "fast open"]

test_x = load("test_x")
test_y = load("test_y")

sample_x = test_x[:100]

clfs = []
for i in range(5):
    clfs.append(load(f"rwguard_model{i}.joblib"))

if any(clf is None for clf in clfs) or test_x is None or test_y is None:
    print("please run train_rwguard.py before running this")
    exit()

preds = []
explanations = []
shap_values = []

for i in range(5):
    preds.append(clfs[i].predict(sample_x))
    explainer = shap.TreeExplainer(clfs[i])
    explanations.append(explainer(sample_x))
    explanations[i].feature_names = feature_names
    shap_values.append(explanations[i].values)


for i in range(5):
    shap.plots.beeswarm(explanations[i][:,:,1])


