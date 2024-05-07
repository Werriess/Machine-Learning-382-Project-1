import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from category_encoders import OneHotEncoder

def prepare_data(data):
    df_raw_new = pd.read_csv(data)
    
    if 'Loan_ID' in df_raw_new.columns:
        df_raw_new.drop(columns = 'Loan_ID')

    missing_values = (
        df_raw_new.isnull().sum()/len(df_raw_new)*100
    ).round(0).astype(int)

    for col in df_raw_new.columns:
        if df_raw_new[col].dtype == 'int64' or df_raw_new[col].dtype == 'float64':
            df_raw_new[col].fillna(df_raw_new[col].mean(), inplace=True)
        else:
            df_raw_new[col].fillna(df_raw_new[col].mode(), inplace=True)

    min_mask = lambda col, val: df_raw_new[col] < val
    income_mask = min_mask('ApplicantIncome',10000)
    coapplicant_income_mask = min_mask('CoapplicantIncome',5701)
    loan_ammount_mask = min_mask('LoanAmount',260)

    df_raw_new1 = df_raw_new[income_mask & coapplicant_income_mask & loan_ammount_mask]

    if 'Loan_Status' in df_raw_new1.columns:
        df_raw_new1['Loan_Status'].apply(lambda s: 1 if s == 'Y' else 0)

    return df_raw_new1  





