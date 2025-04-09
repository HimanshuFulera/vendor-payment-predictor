import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest

# Load dataset
df = pd.read_csv("vendor_payment_data.csv")

# Preprocessing
le_vendor = LabelEncoder()
le_category = LabelEncoder()
le_location = LabelEncoder()
df["Vendor ID"] = le_vendor.fit_transform(df["Vendor ID"])
df["Vendor Category"] = le_category.fit_transform(df["Vendor Category"])
df["Vendor Location"] = le_location.fit_transform(df["Vendor Location"])

# Feature engineering
df["Terms_Amount_Interaction"] = df["Payment Terms"] * df["Payment Amount"]
df["Amount_Category"] = df["Payment Amount"] * df["Vendor Category"]

# Features and target
features = [
    "Vendor ID",
    "Payment Amount",
    "Vendor Category",
    "Vendor Location",
    "Historical Payment Behavior",
    "Payment Terms",
    "Terms_Amount_Interaction",
    "Amount_Category"
]
X = df[features]
y = df["delay_days"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.75,
    colsample_bytree=0.75,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Predict
y_pred = xgb_model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("R² Score:", r2)

# Feature importance
print("Feature Importances:", dict(zip(features, xgb_model.feature_importances_)))

# Anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df["is_anomaly"] = iso_forest.fit_predict(X)

# Save models and encoders
pickle.dump(xgb_model, open("rf_model.pkl", "wb"))
pickle.dump(iso_forest, open("iso_forest.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_vendor, open("le_vendor.pkl", "wb"))
pickle.dump(le_category, open("le_category.pkl", "wb"))
pickle.dump(le_location, open("le_location.pkl", "wb"))

df.to_csv("vendor_payment_data_with_anomalies.csv", index=False)