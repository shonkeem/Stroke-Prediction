import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import preprocessing
'''
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient
'''

X = pd.DataFrame()
y = pd.DataFrame()

datasetpath = ".\healthcare-dataset-stroke-data.csv"
X = pd.read_csv(datasetpath)
y = X['stroke']
X = X.drop(columns=['id', 'stroke'])

le = preprocessing.LabelEncoder()
for column in X.columns:
    if X[column].dtype != np.int64 and X[column].dtype != np.float64:
        classes = set(X[column])
        options = []
        for item in classes:
            options.append(item)
        le.fit(options)
        X[column] = le.transform(X[column])

def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            i = 0 
            for element in unique_elements:
                if element not in text_digit_vals:
                    text_digit_vals[element] = i
                    i += 1
            df[column] = list(map(convert_to_int, df[column]))

# handle_non_numerical_data(X)

imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
CleanX = pd.DataFrame(imp.fit_transform(X))
CleanX.columns = X.columns
CleanX.index = X.index

train_X, validation_X, train_y, validation_y = train_test_split(
    CleanX, y, random_state=1, test_size=0.2)