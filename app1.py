import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import os
import requests  # add this import!

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
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
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

FILE_ID = "1TKWI5I7D32YOFi3oQpQlV5kPz9jBRJ8K"
MODEL_PATH = "monthly_income_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_file_from_google_drive(FILE_ID, MODEL_PATH)
    print("Download complete.")

model = joblib.load(MODEL_PATH)

# Load other objects
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')
data = pd.read_csv('train.csv')

# Your Streamlit UI code remains unchanged...


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
