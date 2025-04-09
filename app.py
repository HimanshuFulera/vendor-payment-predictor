from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
rf_model = pickle.load(open("rf_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le_vendor = pickle.load(open("le_vendor.pkl", "rb"))
le_category = pickle.load(open("le_category.pkl", "rb"))
le_location = pickle.load(open("le_location.pkl", "rb"))
iso_forest = pickle.load(open("iso_forest.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print("POST request received")
        # Get form data
        vendor_id = request.form["vendor_id"]
        payment_amount = float(request.form["payment_amount"])
        vendor_category = request.form["vendor_category"]
        vendor_location = request.form["vendor_location"]
        hist_payment_behavior = float(request.form["hist_payment_behavior"])
        invoice_date = request.form["invoice_date"]
        due_date = request.form["due_date"]

        print(f"Input: {vendor_id}, {payment_amount}, {vendor_category}, {vendor_location}, "
              f"{hist_payment_behavior}, {invoice_date}, {due_date}")

        # Calculate Payment Terms
        try:
            invoice_dt = datetime.strptime(invoice_date, "%Y-%m-%d")
            due_dt = datetime.strptime(due_date, "%Y-%m-%d")
            payment_terms = (due_dt - invoice_dt).days
            print(f"Payment Terms: {payment_terms}")
        except ValueError as e:
            print(f"Date parsing error: {e}")
            return render_template("index.html", prediction="Error: Invalid date format", 
                                 anomaly=None, hist_data=None)

        # Encode categorical variables
        try:
            vendor_id_enc = le_vendor.transform([vendor_id])[0]
            category_enc = le_category.transform([vendor_category])[0]
            location_enc = le_location.transform([vendor_location])[0]
            print(f"Encoded: {vendor_id_enc}, {category_enc}, {location_enc}")
        except ValueError as e:
            print(f"Encoding error: {e}")
            return render_template("index.html", prediction="Error: Invalid category/location/vendor", 
                                 anomaly=None, hist_data=None)

        # Prepare input data
        input_data = np.array([[vendor_id_enc, payment_amount, category_enc, location_enc, 
                                hist_payment_behavior, payment_terms, 
                                payment_terms * payment_amount, payment_amount * category_enc]])
        input_scaled = scaler.transform(input_data)

        # Predict delay
        predicted_delay = rf_model.predict(input_scaled)[0]
        is_anomaly = iso_forest.predict(input_data)[0] == -1
        print(f"Predicted Delay: {predicted_delay}, Anomaly: {is_anomaly}")

        # Generate historical data based on input
        hist_data = [max(0, hist_payment_behavior + np.random.normal(0, 2)) for _ in range(4)]  # Simulate past delays

        # Pass form data back to template
        return render_template("index.html", 
                              prediction=f"Predicted Delay: {predicted_delay:.2f} days",
                              anomaly="Yes" if is_anomaly else "No",
                              hist_data=hist_data,
                              pred_delay=predicted_delay,
                              vendor_id=vendor_id,
                              payment_amount=payment_amount,
                              vendor_category=vendor_category,
                              vendor_location=vendor_location,
                              hist_payment_behavior=hist_payment_behavior,
                              invoice_date=invoice_date,
                              due_date=due_date)
    return render_template("index.html", prediction=None, anomaly=None, hist_data=None)

if __name__ == "__main__":
    app.run(debug=True)