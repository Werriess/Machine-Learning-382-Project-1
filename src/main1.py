from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
import joblib

df = prepare_data('../data/raw_data.csv')
df.drop(columns = 'Married_nan', inplace= True)

X = df.drop(['Loan_Status'], axis=1)
Y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_model():
    model = Sequential([
        Dense(25, activation='relu', input_shape=(25,)),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
model = create_model()  
model.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_data=(X_test_scaled, y_test))

joblib.dump(model, '../artifacts/model1.pkl')

model = joblib.load('../model1.pkl')

datav = prepare_data('../data/validation.csv')

X_val_scaled = scaler.transform(datav)

predictions = model.predict(X_val_scaled)

predictions_arr = []

for status in predictions:
    if status > 0.5:
        predictions_arr.append('Yes')
    else:
        predictions_arr.append('No')

print(predictions_arr)