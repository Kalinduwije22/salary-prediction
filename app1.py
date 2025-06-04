import os
import requests
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Google Drive download helper functions
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

# Validate if file is a pickle file (basic check)
def is_valid_pickle(filepath):
    try:
        with open(filepath, "rb") as f:
            header = f.read(2)
            return header == b'\x80\x04'  # Pickle protocol header bytes
    except Exception:
        return False

# File info
MODEL_FILE_ID = "1TKWI5I7D32YOFi3oQpQlV5kPz9jBRJ8K"
MODEL_PATH = "monthly_income_model.pkl"

# Streamlit UI start
st.title("Employee Monthly Income Prediction")

# Download model if needed and validate
if not os.path.exists(MODEL_PATH) or not is_valid_pickle(MODEL_PATH):
    st.info("Downloading model from Google Drive...")
    try:
        download_file_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()
    if not is_valid_pickle(MODEL_PATH):
        st.error("Downloaded model file is invalid or corrupted.")
        st.stop()
    st.success("Model downloaded successfully.")

# Check other required files
required_files = ['scaler.pkl', 'columns.pkl', 'train.csv']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file '{file}' not found. Please upload it to your project folder.")
        st.stop()

# Load model and related files safely
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    data = pd.read_csv('train.csv')
except Exception as e:
    st.error(f"Failed to load model or related files: {e}")
    st.stop()

# Prepare input data
selected_columns = ['Gender', 'Years at Company', 'Job Role', 'Job Level', 'Company Size', 'Age']
X = data[selected_columns]
cat_cols = ['Gender', 'Job Role', 'Job Level', 'Company Size', 'Age']

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

    # Encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X[col])
        input_df[col] = le.transform(input_df[col])

    # Add missing columns with default 0
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training
    input_df = input_df[columns]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict income
    predicted_income = model.predict(input_scaled)[0]
    st.success(f"Predicted Monthly Income: {predicted_income:.2f}")
