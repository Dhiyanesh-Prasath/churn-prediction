import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    df = df.drop(['rownumber', 'customerid', 'surname'], axis=1)

    categorical_cols = ['geography', 'gender']
    
    numerical_cols = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 
                      'hascrcard', 'isactivemember', 'estimatedsalary']
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = SimpleImputer(strategy='mean')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X = df.drop('churn', axis=1)
    y = df['churn']

    return X, y, preprocessor