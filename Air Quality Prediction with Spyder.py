# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load your data into a DataFrame
data = pd.read_csv('data/city_day.csv', na_values='=')

# Fill missing values with means for numeric columns
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Feature selection
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
data = data[features]

# Split data into features and labels
X = data.drop(columns=['AQI'])
y = data['AQI']

# Train a RandomForestRegressor model
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

# Create a Streamlit app
st.title('Air Quality Prediction System')

# Create columns for input fields
input_col1, input_col2, input_col3 = st.columns(3)

# Input fields in the first column
with input_col1:
    st.write("Input Features")
    pm25 = st.number_input('PM2.5', min_value=0.0)
    pm10 = st.number_input('PM10', min_value=0.0)
    no = st.number_input('NO', min_value=0.0)
    no2 = st.number_input('NO2', min_value=0.0)

# Input fields in the second column
with input_col2:
    nox = st.number_input('NOx', min_value=0.0)
    nh3 = st.number_input('NH3', min_value=0.0)
    co = st.number_input('CO', min_value=0.0)
    so2 = st.number_input('SO2', min_value=0.0)

# Input fields in the third column
with input_col3:
    o3 = st.number_input('O3', min_value=0.0)
    benzene = st.number_input('Benzene', min_value=0.0)
    toluene = st.number_input('Toluene', min_value=0.0)
    xylene = st.number_input('Xylene', min_value=0.0)

# Make a prediction using the trained model
input_data = [[pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene]]
predicted_aqi = regr.predict(input_data)[0]

# Display the predicted AQI
st.write('Predicted AQI:', predicted_aqi)
