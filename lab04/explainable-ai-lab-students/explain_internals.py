import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from sklearn import metrics
import csv
import pandas as pd

clf = load("rwguard_model.joblib")
test_x = load("test_x")
test_y = load("test_y")

sample_x = test_x[:100]

if clf is None or test_x is None or test_y is None:
    print("please run train_rwguard.py before running this")
    exit()

pred = clf.predict(sample_x)


explainer = shap.TreeExplainer(clf)
explanation = explainer(sample_x)

shap_values = explanation.values
# make sure the SHAP values add up to marginal predictions

shap.plots.beeswarm(explanation[:,:,1])