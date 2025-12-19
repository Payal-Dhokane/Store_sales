from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/sales_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "store_nbr": int(request.form["store_nbr"]),
            "family": int(request.form["family"]),
            "onpromotion": int(request.form["onpromotion"]),
            "day": int(request.form["day"]),
            "month": int(request.form["month"])
        }

        X_new = pd.DataFrame([data])
        pred_log = model.predict(X_new)
        pred_sales = np.expm1(pred_log)

        return render_template(
            "index.html",
            prediction=f"Predicted Sales: {pred_sales[0]:.2f}"
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
