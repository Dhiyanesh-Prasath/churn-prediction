import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data(df):
    df = df.drop(['id'], axis=1)

    numerical_cols = ['subscription_age', 'bill_avg', 'reamining_contract', 'service_failure_count',
                      'download_avg', 'upload_avg', 'download_over_limit']
    
    categorical_cols = ['is_tv_subscriber', 'is_movie_package_subscriber']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = SimpleImputer(strategy='most_frequent')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X = df.drop('churn', axis=1)
    y = df['churn']

    return X, y, preprocessor

def test_model_metrics(model_path, data_path):
    df = pd.read_csv(data_path)
    
    X, y, preprocessor = preprocess_data(df)
    
    X_preprocessed = preprocessor.fit_transform(X)
    
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
    model_path = 'api/models/internet_service_churn_model.pkl'
    data_path = 'data/internet_service_churn.csv'
    print(f"Metrics for Internet Service Churn model ({data_path}):")
    
    test_model_metrics(model_path, data_path)