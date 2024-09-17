from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from models.model import load_model
import joblib

app = Flask(__name__)

model_1 = load_model("api/models/Bank_churn_model.pkl")
preprocessor_1 = joblib.load("api/preprocessors/preprocessor_1.pkl")

model_2 = load_model("api/models/internet_service_churn_model.pkl")
preprocessor_2 = joblib.load("api/preprocessors/preprocessor_2.pkl")

columns_1 = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 
             'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary']

columns_2 = ['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age', 
             'bill_avg', 'reamining_contract', 'service_failure_count', 
             'download_avg', 'upload_avg', 'download_over_limit']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features_list = data['features']

    predictions = []

    for features in features_list:
        if len(features) == len(columns_1):
            features = np.array(features).reshape(1, -1)
            X_new = pd.DataFrame(features, columns=columns_1)
            X_new_preprocessed = preprocessor_1.transform(X_new)
            prediction = model_1.predict(X_new_preprocessed)[0]
            predictions.append(int(prediction))

        elif len(features) == len(columns_2):
            features = np.array(features).reshape(1, -1)
            X_new = pd.DataFrame(features, columns=columns_2)
            X_new_preprocessed = preprocessor_2.transform(X_new)
            prediction = model_2.predict(X_new_preprocessed)[0]
            predictions.append(int(prediction))

        else:
            predictions.append("Invalid feature length")

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True, port=5000)