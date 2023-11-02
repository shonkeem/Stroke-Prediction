from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve
from StrokePrediction import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(train_X, train_y)
predictions = LR_model.predict(validation_X)

probs = LR_model.predict_proba(validation_X)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(validation_y, probs)

print(roc_auc_score(validation_y, probs))
# print(classification_report(validation_y, predictions))

plt.plot(fpr, tpr, color='r', label="LR")
