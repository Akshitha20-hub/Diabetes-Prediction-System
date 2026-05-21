# ==========================================
# Diabetes Prediction Using Logistic Regression
# ==========================================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Step 2: Load Dataset
# ==========================================

# Make sure diabetes.csv is in the same folder
data = pd.read_csv("diabetes.csv")

# ==========================================
# Step 3: Display Dataset
# ==========================================

print("========== FIRST 5 ROWS ==========")
print(data.head())

# ==========================================
# Step 4: Dataset Information
# ==========================================

print("\n========== DATASET INFORMATION ==========")
print(data.info())

print("\n========== DATASET DESCRIPTION ==========")
print(data.describe())

# ==========================================
# Step 5: Check Missing Values
# ==========================================

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# ==========================================
# Step 6: Data Cleaning
# Replace 0 values with Mean
# ==========================================

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
    data[col] = data[col].replace(0, data[col].mean())

print("\nData Cleaning Completed Successfully")

# ==========================================
# Step 7: Separate Features and Target
# ==========================================

# X = Input Features
X = data.drop("Outcome", axis=1)

# y = Output
y = data["Outcome"]

# ==========================================
# Step 8: Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Completed")
print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# ==========================================
# Step 9: Create Logistic Regression Model
# ==========================================

model = LogisticRegression(max_iter=1000)

# ==========================================
# Step 10: Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully")

# ==========================================
# Step 11: Test Model
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Step 12: Check Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy:", accuracy)

# ==========================================
# Step 13: Predict New Patient
# ==========================================

# Patient Data Format:
# [Pregnancies, Glucose, BloodPressure,
#  SkinThickness, Insulin, BMI,
#  DiabetesPedigreeFunction, Age]

new_patient = np.array([[2, 180, 75, 30, 120, 28, 0.5, 45]])

prediction = model.predict(new_patient)

print("\n========== PREDICTION RESULT ==========")

if prediction[0] == 1:
    print("Diabetes: Yes")
else:
    print("Diabetes: No")

# ==========================================
# Step 14: Final Message
# ==========================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("Diabetes Prediction Model is Ready")
print("===================================")