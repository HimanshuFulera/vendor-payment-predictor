import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np

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

# Anomaly detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
# Fit on training data to avoid warning, predict on test data
iso_forest.fit(X_train_scaled)
anomaly_preds = iso_forest.predict(X_test_scaled)

# Save models and encoders
pickle.dump(xgb_model, open("rf_model.pkl", "wb"))
pickle.dump(iso_forest, open("iso_forest.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(le_vendor, open("le_vendor.pkl", "wb"))
pickle.dump(le_category, open("le_category.pkl", "wb"))
pickle.dump(le_location, open("le_location.pkl", "wb"))

df["is_anomaly"] = iso_forest.predict(scaler.transform(X))  # Apply to full dataset
df.to_csv("vendor_payment_data_with_anomalies.csv", index=False)

# Graph 1: Delay Prediction vs. Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # 45-degree line
plt.xlabel('Actual Delay Days')
plt.ylabel('Predicted Delay Days')
plt.title(f'Delay Prediction vs. Actual (XGBoost, R² = {r2:.4f}, MAE = {mae:.4f})')
plt.tight_layout()
plt.savefig('delay_prediction_vs_actual.png')
plt.show()  # Display the graph after saving

# Graph 2: Anomaly Detection Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, c=anomaly_preds, cmap='coolwarm', alpha=0.6)
plt.xlabel('Test Sample Index')
plt.ylabel('Delay Days')
plt.title('Anomaly Detection by Isolation Forest')
plt.colorbar(label='Anomaly (-1) vs Normal (1)')
plt.tight_layout()
plt.savefig('anomaly_detection_scatter.png')
plt.show()  # Display the graph after saving

# Graph 3: Feature Importance Bar Chart
feature_importance = xgb_model.feature_importances_
feature_names = features
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance[:5])), feature_importance[:5], align='center')
plt.yticks(range(len(feature_importance[:5])), feature_names[:5])
plt.xlabel('Feature Importance')
plt.title('Top 5 Feature Importances (XGBoost)')
plt.tight_layout()
plt.savefig('feature_importance_bar.png')
plt.show()  # Display the graph after saving

# Graph 4: Train-Test Split Performance
y_train_pred = xgb_model.predict(X_train_scaled)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mae = mae
test_r2 = r2

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = np.arange(2)
plt.bar(x - bar_width/2, [train_r2, test_r2], bar_width, color=['blue', 'orange'], alpha=0.7, label='R2 Score')
plt.ylabel('R2 Score')
plt.title('Train vs Test Performance')
plt.ylim(0, 1)
plt.xticks(x, ['Train', 'Test'])
plt.legend()
plt.twinx()
plt.bar(x + bar_width/2, [train_mae, test_mae], bar_width, color=['green', 'red'], alpha=0.7, label='MAE')
plt.ylabel('MAE')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('train_test_split_performance.png')
plt.show()  # Display the graph after saving

print("Graphs saved as PNG files: delay_prediction_vs_actual.png, anomaly_detection_scatter.png, feature_importance_bar.png, train_test_split_performance.png")
