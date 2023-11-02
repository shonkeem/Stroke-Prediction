from StrokePrediction import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

RF_model = RandomForestClassifier()
RF_model.fit(train_X, train_y)
predictions = RF_model.predict(validation_X)


probs = RF_model.predict_proba(validation_X)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(validation_y, probs)

# print(classification_report(validation_y, predictions))
print(roc_auc_score(validation_y, probs))
plt.plot(fpr, tpr, color='b', label="RF")
