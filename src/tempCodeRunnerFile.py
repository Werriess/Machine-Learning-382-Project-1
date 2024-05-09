from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from prepare_data import prepare_data
from sklearn.model_selection import train_test_split
import joblib

df = prepare_data('../MLG382 Projects/Machine-Learning-382-Project-1/data/raw_data.csv')
print(df.head())