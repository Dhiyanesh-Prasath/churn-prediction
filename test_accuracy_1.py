import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data_bank_churn(df, preprocessor):
    df = df.drop(['rownumber', 'customerid', 'surname'], axis=1)

    X = df.drop('churn', axis=1)
    y = df['churn']

    X_preprocessed = preprocessor.transform(X)
    
    return X_preprocessed, y

def test_model_metrics(model_path, preprocessor_path, data_path):
    df = pd.read_csv(data_path)
    
    preprocessor = joblib.load(preprocessor_path)
    
    X_preprocessed, y = preprocess_data_bank_churn(df, preprocessor)
    
    model = joblib.load(model_path)
    
    y_pred = model.predict(X_preprocessed)
    y_proba = model.predict_proba(X_preprocessed)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC-ROC: {roc_auc:.4f}')

if __name__ == "__main__":
    model_path = 'api/models/Bank_churn_model.pkl'
    preprocessor_path = 'api/preprocessors/preprocessor_1.pkl'
    data_path = 'data/Bank_churn.csv'
    print(f"Metrics for Bank Churn Model ({data_path}):")
    
    test_model_metrics(model_path, preprocessor_path, data_path)