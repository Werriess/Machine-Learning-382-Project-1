import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from scikeras.wrappers import KerasClassifier



df_raw = pd.read_csv('C:/Users/werne/OneDrive/Desktop/Projects/Python/Project 1 MLG/data/raw_data.csv')
df_test = pd.read_csv('C:/Users/werne/OneDrive/Desktop/Projects/Python/Project 1 MLG/data/validation.csv')

df_raw_new = df_raw.drop(columns = 'Loan_ID')

missing_values = (
    df_raw_new.isnull().sum()/len(df_raw_new)*100
).round(0).astype(int)

print(f'Column\t\t\t% missing')
print(f'{"-"}'*35)
missing_values.round(2)

for col in df_raw_new.columns:
    if df_raw_new[col].dtype == 'int64' or df_raw_new[col].dtype == 'float64':
        df_raw_new[col].fillna(df_raw_new[col].mean(), inplace=True)
    else:
        df_raw_new[col].fillna(df_raw_new[col].mode(), inplace=True)


df_raw_new1 = df_raw_new[df_raw_new['ApplicantIncome'] < 10000]
df_raw_new2 = df_raw_new1[df_raw_new1['CoapplicantIncome'] < 5701]
df_raw_new3 = df_raw_new2[df_raw_new2['LoanAmount'] < 260]


df_raw_new3['Loan_Status'] = df_raw_new3['Loan_Status'].apply(lambda s: 1 if s == 'Y' else 0)


ohe = OneHotEncoder(
    use_cat_names=True, 
    cols=['Gender', 'Dependents', 'Married', 'Education', 'Self_Employed', 'Property_Area']
)

# Transform data
encoded_df = ohe.fit_transform(df_raw_new3)


print(encoded_df)

X = encoded_df.drop(['Loan_Status'], axis=1) 
Y = encoded_df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(24, input_shape=(24,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# # evaluate model with standardized dataset
# estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


...
#evaluate baseline model with standardized dataset
estimators = []
estimators.append(('onehotencoder', OneHotEncoder(use_cat_names=True)))
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# Standardized: 77.90% (5.24%)
#Standardized: 78.62% (4.35%)
#Standardized: 76.57% (4.82%) with onehotencoder added
#Standardized: 76.94% (3.62%) with onehotencoder added


# smaller model
# def create_smaller():
#     # create model
#     model = Sequential()
#     model.add(Dense(12, input_shape=(24,), activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(model=create_smaller, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#larger model
# def create_larger():
# # create model
#     model = Sequential()
#     model.add(Dense(64, input_shape=(24,), activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
# estimators = []
# estimators.append(('onehotencoder', OneHotEncoder(use_cat_names=True)))
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(model=create_larger, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Larger: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#Feature engineering
#df_raw_new3['TotalIncome'] = df_raw_new3['ApplicantIncome'] + df_raw_new3['CoapplicantIncome']