import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load model and preprocessing objects
model = joblib.load('monthly_income_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')  # All columns used during training
data = pd.read_csv('train.csv')

# Use only selected features for input
selected_columns = ['Gender', 'Years at Company', 'Job Role', 'Job Level', 'Company Size','Age']
X = data[selected_columns]
cat_cols = ['Gender', 'Job Role', 'Job Level', 'Company Size','Age']
st.title("Employee Monthly Income Prediction")

st.write("Enter the following employee details:")

# Collect user input
input_data = {}
for col in selected_columns:
    if X[col].dtype == 'object':
        options = sorted(X[col].dropna().unique().tolist())
        input_data[col] = st.selectbox(col, options)
    else:
        min_val = int(X[col].min())
        max_val = int(X[col].max())
        mean_val = int(X[col].median())
        input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

if st.button("Predict"):
    # Create DataFrame with selected features
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X[col])
        input_df[col] = le.transform(input_df[col])

    # Fill missing columns with default value (e.g., 0)
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure column order matches training
    input_df = input_df[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    predicted_income = model.predict(input_scaled)[0]
    st.success(f"Predicted Monthly Income: {predicted_income:.2f}")
