from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

df = prepare_data('./Project 1 MLG/data/raw_data.csv')

X = df.drop(['Loan_Status'], axis=1) 
Y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(24, input_shape=(24,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# evaluate model with standardized dataset
# estimator = KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)
# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#evaluate baseline model with standardized dataset
def cpipeline():
    estimators = []
    estimators.append(('onehotencoder', OneHotEncoder(use_cat_names=True)))
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

    #kfold = StratifiedKFold(n_splits=10, shuffle=True)
    #results = cross_val_score(pipeline, X, Y, cv=kfold)
    #print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

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