import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data_2(df):
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