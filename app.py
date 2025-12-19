
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load trained LightGBM model
model = joblib.load('model/sales_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_new = pd.DataFrame([data])
        pred_log = model.predict(X_new)
        pred_sales = np.expm1(pred_log)
        return jsonify({'predicted_sales': float(pred_sales[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
