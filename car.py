# ==========================================
# Car Price Predictor (100% Error-Free Final)
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("quikr_car.csv")

# ==========================================
# Data Cleaning
# ==========================================

# Remove invalid prices
data = data[data['Price'] != 'Ask For Price']
data['Price'] = data['Price'].str.replace(',', '').astype(int)

# 🔥 FINAL FIX FOR kms_driven (NO ERROR)
data['kms_driven'] = data['kms_driven'].astype(str)
data['kms_driven'] = data['kms_driven'].str.replace(',', '')
data['kms_driven'] = data['kms_driven'].str.replace(' kms', '')

# Convert safely
data['kms_driven'] = pd.to_numeric(data['kms_driven'], errors='coerce')

# Drop NaN values
data = data.dropna(subset=['kms_driven'])

data['kms_driven'] = data['kms_driven'].astype(int)

# Clean year
data = data[data['year'].astype(str).str.isnumeric()]
data['year'] = data['year'].astype(int)

# Drop remaining nulls
data = data.dropna()

# ==========================================
# Features & Target
# ==========================================

X = data[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = data['Price']

# ==========================================
# Preprocessing + Model
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         ['name', 'company', 'fuel_type'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# ==========================================
# Train Model
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Evaluation
# ==========================================

y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# ==========================================
# Show Options
# ==========================================

print("\nAvailable Companies:")
print(data['company'].unique())

print("\nAvailable Fuel Types:")
print(data['fuel_type'].unique())

print("\nSome Car Names:")
print(data['name'].unique()[:20])

# ==========================================
# User Input
# ==========================================

print("\nEnter car details:")

name = input("Car Name (copy from above): ")
company = input("Company (copy from above): ")
year = int(input("Year: "))
kms = int(input("KMs Driven: "))
fuel = input("Fuel Type (copy from above): ")

# ==========================================
# Prediction
# ==========================================

input_data = pd.DataFrame([[name, company, year, kms, fuel]],
                          columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

prediction = model.predict(input_data)

print("\nEstimated Price: ₹", int(prediction[0]))