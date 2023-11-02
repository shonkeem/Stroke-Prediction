from ast import increment_lineno
import pandas as pd
from StrokePrediction import *

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
datasetpath = ".\healthcare-dataset-stroke-data.csv"
X = pd.read_csv(datasetpath)
y = X['stroke']
# print(X)
# print(X.loc[X['stroke'].isin([1])])
# print(len(X.loc[X['stroke'].isin([1])]))
# print(X.head())
# print(y.head())

plt.figure(figsize=(16,6))
sns.lineplot(data=X)