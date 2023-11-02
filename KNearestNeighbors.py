from asyncio import as_completed
from StrokePrediction import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

knn_model = KNeighborsClassifier()
knn_model.fit(train_X, train_y)
predictions = knn_model.predict(validation_X)

probs = knn_model.predict_proba(validation_X)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(validation_y, probs)

print(roc_auc_score(validation_y, probs))
# print(classification_report(validation_y, predictions))

plt.plot(fpr, tpr, color='g', label="KNN")
