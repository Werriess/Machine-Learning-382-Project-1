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