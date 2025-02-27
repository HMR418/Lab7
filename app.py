# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import streamlit as st

# Read the dataset from the local file in the GitHub repository
df = pd.read_excel('AmesHousing.xlsx')

# --- Data Preparation ---
# Use only the features that are numeric and match our app inputs.
features = ['GrLivArea', 'OverallQual']
X = df[features]
y = df['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train the Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# (Optional) Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'MSE: {mse}, R²: {r2}')

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# --- Streamlit Web Application ---
# Load the model for the app
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Ames Housing Price Predictor")

# User input widgets for features used in the model
gr_liv_area = st.number_input('Above Ground Living Area (sq ft):', min_value=0, value=1500)
overall_qual = st.number_input('Overall Quality (1-10):', min_value=1, max_value=10, value=5)

if st.button('Predict'):
    # Construct a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'GrLivArea': [gr_liv_area],
        'OverallQual': [overall_qual]
    })
    # Make a prediction and display the result
    prediction = model.predict(input_data)
    st.write(f"Predicted Sale Price: ${prediction[0]:,.2f}")
