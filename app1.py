import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import os
import requests

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download&id=1TKWI5I7D32YOFi3oQpQlV5kPz9jBRJ8K"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # Also check in the HTML content for token if not found in cookies
    if "confirm=" in response.text:
        import re
        matches = re.findall(r'confirm=([0-9A-Za-z_]+)', response.text)
        if matches:
            return matches[0]
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Your Google Drive FILE_ID of the model
FILE_ID = "1TKWI5I7D32YOFi3oQpQlV5kPz9jBRJ8K"
MODEL_PATH = "monthly_income_model.pkl"

st.title("Employee Monthly Income Prediction")

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    try:
        download_file_from_google_drive(FILE_ID, MODEL_PATH)
        st.success("Model download complete.")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# Load model and other objects with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    data = pd.read_csv('train.csv')
except Exception as e:
    st.error(f"Failed to load model or preprocessing files: {e}")
    st.stop()

# Selected features for input
selected_columns = ['Gender', 'Years at Company', 'Job Role', 'Job Level', 'Company Size', 'Age']
X = data[selected_columns]
cat_cols = ['Gender', 'Job Role', 'Job Level', 'Company Size', 'Age']

st.write("Enter the following employee details:")

# Collect user input dynamically based on feature type
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
    try:
        # Create dataframe from input
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns with LabelEncoder fit on training data
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X[col])
            input_df[col] = le.transform(input_df[col])

        # Add any missing columns with zeros
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Ensure order of columns matches training
        input_df = input_df[columns]

        # Scale inputs
        input_scaled = scaler.transform(input_df)

        # Make prediction
        predicted_income = model.predict(input_scaled)[0]
        st.success(f"Predicted Monthly Income: {predicted_income:.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
