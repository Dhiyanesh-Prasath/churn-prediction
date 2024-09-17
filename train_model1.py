import pandas as pd
from sklearn.model_selection import train_test_split
from models.model import train_random_forest_model, save_model
from preprocessing.preprocess1 import preprocess_data
import joblib

df = pd.read_csv('data/Bank_churn.csv')

X, y, preprocessor = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor.fit(X_train)

X_train_preprocessed = preprocessor.transform(X_train)

rf_model = train_random_forest_model(X_train_preprocessed, y_train)

save_model(rf_model, 'api/models/Bank_churn_model.pkl')

joblib.dump(preprocessor, 'api/preprocessors/preprocessor_1.pkl')