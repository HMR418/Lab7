import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Set working directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
st.write("Current working directory:", os.getcwd())

# Read the dataset from the local Excel file
try:
    df = pd.read_excel('AmesHousing.xlsx')
except Exception as e:
    st.error("Error reading 'AmesHousing.xlsx': " + str(e))
    st.stop()

# Remove any extra whitespace from column names
df.columns = df.columns.str.strip()

# Display the column names for verification
st.write("Columns in dataset:", df.columns.tolist())

# Check if required columns are present
required_columns = ['GrLivArea', 'OverallQual', 'SalePrice']
if not all(col in df.columns for col in required_columns):
    st.error("One or more required columns are missing. Required columns: " + str(required_columns))
    st.stop()

# Use only the features that match our app inputs
features = ['GrLivArea', 'OverallQual']
X = df[features]
y = df['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# (Optional) Evaluate the model and display metrics
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
st.write(f'Model Evaluation: MSE = {mse:.2f}, R² = {r2:.2f}')

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the model for the app (in production, you might separate training and prediction)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# --- Streamlit Web Application Interface ---
st.title("Ames Housing Price Predictor")

# User input widgets for the features
gr_liv_area = st.number_input('Above Ground Living Area (sq ft):', min_value=0, value=1500)
overall_qual = st.number_input('Overall Quality (1-10):', min_value=1, max_value=10, value=5)

if st.button('Predict'):
    # Construct input DataFrame matching the training format
    input_data = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'OverallQual': [overall_qual]
    })
    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Predicted Sale Price: ${prediction[0]:,.2f}")
