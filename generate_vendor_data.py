import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)

# Parameters
num_records = 7000
vendors = [f"V{i:03d}" for i in range(1, 101)]
categories = ["Logistics", "Raw Materials", "Services", "Technology"]
locations = ["US", "EU", "Asia", "Africa"]

# Generate dataset
data = {
    "Vendor ID": [random.choice(vendors) for _ in range(num_records)],
    "Invoice Date": [datetime(2023, 1, 1) + timedelta(days=random.randint(0, 730)) for _ in range(num_records)],
    "Payment Amount": [random.randint(500, 75000) for _ in range(num_records)],
    "Vendor Category": [random.choice(categories) for _ in range(num_records)],
    "Vendor Location": [random.choice(locations) for _ in range(num_records)],
}

df = pd.DataFrame(data)

# Generate Due Date with variable terms
payment_terms_days = np.random.randint(7, 121, num_records)
df["Due Date"] = df["Invoice Date"] + pd.to_timedelta(payment_terms_days, unit="D")
df["Payment Terms"] = (df["Due Date"] - df["Invoice Date"]).dt.days

# Realistic delay: Stronger feature influence
category_delay_factor = {"Logistics": 2.5, "Raw Materials": 1.8, "Services": 1.2, "Technology": 1.5}
location_delay_factor = {"US": 1.2, "EU": 1.5, "Asia": 1.8, "Africa": 2.5}
terms_delay = (120 - df["Payment Terms"]) / 20  # Max 6 days
amount_delay = np.log1p(df["Payment Amount"]) / np.log1p(75000) * 5  # Max 5 days
base_noise = np.random.normal(0, 1, num_records)  # Reduced noise

# Calculate delay tendency
delay_tendency = (
    base_noise +
    terms_delay * 1.5 +  # Amplify terms impact
    amount_delay +
    df["Vendor Category"].map(category_delay_factor) +
    df["Vendor Location"].map(location_delay_factor)
)

# Assign payment timing: ~25% early, 25% on-time, 50% late
payment_timing = np.where(
    delay_tendency < 0.5, "early",  # ~25%
    np.where(delay_tendency < 2.5, "on_time", "late")  # ~25% on-time, ~50% late
)

# Apply rules
early_shift = np.random.randint(-5, 0, num_records)
df["delay_days"] = np.where(
    payment_timing == "early", 0,
    np.where(
        payment_timing == "on_time", 0,
        delay_tendency.clip(1, 20)
    )
).astype(int)

# Calculate Payment Date
df["Payment Date"] = np.where(
    payment_timing == "early",
    df["Due Date"] + pd.to_timedelta(early_shift, unit="D"),
    df["Due Date"] + pd.to_timedelta(df["delay_days"], unit="D")
)
df["is_delayed"] = (df["delay_days"] > 3).astype(int)

# Sort and calculate Historical Payment Behavior with stronger tie-in
df = df.sort_values(["Vendor ID", "Invoice Date"]).reset_index(drop=True)
df["Historical Payment Behavior"] = (
    df.groupby("Vendor ID")["delay_days"]
    .shift(1)
    .groupby(df["Vendor ID"])
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
    .fillna(0)
)

# Save to CSV
df.to_csv("vendor_payment_data.csv", index=False)

print("Dataset Shape:", df.shape)
print(df.head())
print("\ndelay_days Stats:")
print(df["delay_days"].describe())
print("\nCount of delay_days values:")
print(df["delay_days"].value_counts().sort_index())
print("\nHistorical Payment Behavior Stats:")
print(df["Historical Payment Behavior"].describe())