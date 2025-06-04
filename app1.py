import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import requests

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

def is_valid_pickle(filepath):
    # Quick check if file looks like a pickle by reading first bytes
    try:
        with open(filepath, "rb") as f:
            header = f.read(2)
            return header == b'\x80\x04'  # pickle protocol header
    except Exception:
        return False

# Google Drive file ID for your model file
FILE_ID = "1TKWI5I7D32YOFi3oQpQlV5kPz9jBRJ8K"
MODEL_PATH = "monthly_income_model.pkl"

# Download model if not exists or invalid
if not os.path.exists(MODEL_PATH) or not is_valid_pickle(MODEL_PATH):
    st.write("Downloading model from Google Drive...")
    download_file_from_google_drive(FILE_ID, MODEL_PATH)
    st.write("Download complete.")

# Load model and other objects safely
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    data = pd.read_csv('train.csv')
except Exception as e:
    st.error(f"Failed to load model or related files: {e}")
    st.stop()

# Prepare Streamlit UI
selected_columns = ['Gender', 'Years at Company', 'Job Role', 'Job Level', 'Company Size', 'Age']
cat_cols = ['Gender', 'Job Role', 'Job Level', 'Company Size', 'Age']

X = data[selected_columns]

st.title("Employee Monthly Income Prediction")
st.write("Enter the following employee details:")

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
    input_df = pd.DataFrame([input_data])

    # Encode categorical features using training data LabelEncoder logic
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X[col])
        try:
            input_df[col] = le.transform(input_df[col])
        except ValueError:
            st.error(f"Invalid value for {col}. Please select a valid option.")
            st.stop()

    # Fill missing columns with 0 (to match training data)
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict and display result
    predicted_income = model.predict(input_scaled)[0]
    st.success(f"Predicted Monthly Income: {predicted_income:.2f}")
