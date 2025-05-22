import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained models
random_forest_model = joblib.load('rating_prediction_model.joblib')

# Title and description
st.title('Coffee Sales Rating Performance Prediction')
st.write(''' This application predicts coffee sales rating based on review and related features. 
Choose the model you want to use for predictions. 
''')

# Sidebar for selecting the model
model_choice = st.sidebar.selectbox(
    'Select a model', ('Random Forest', 'Decision Tree')
)

# Input fields for user data
st.header('Enter Input Data')
roast_dark = st.number_input('Roast Dark', min_value=0.0, max_value=24.0, step=0.1)
roast_light = st.number_input('Roast Light', min_value=0.0, max_value=24.0, step=0.1)
roast_medium = st.number_input('Roast Medium', min_value=0.0, max_value=24.0, step=0.1)
roast_medium_dark = st.number_input('Roast Medium Dark', min_value=0.0, max_value=10.0, step=0.1)
roast_medium_light = st.number_input('Roast Medium Light', min_value=0.0, max_value=1.0, step=0.1)
roast_very_dark = st.number_input('Roast Very Dark', min_value=0.0, max_value=10.0, step=0.1)
region = st.number_input('Region', min_value=0.0, max_value=10.0, step=0.1)

# Prediction
if st.button('Predict'):
    input_data = pd.DataFrame({
        'roast_dark': [roast_dark],
        'roast_light': [roast_light],
        'roast_medium': [roast_medium],
        'roast_medium_dark': [roast_medium_dark],
        'roast_medium_light': [roast_medium_light],
        'roast_very_dark': [roast_very_dark],
        'region': [region] })


    # Choose model for prediction
    if model_choice == 'Random Forest':
        prediction = random_forest_model.predict()[0]
    elif model_choice == 'Decision Tree':
        prediction = decision_tree_model.predict(scaled_data)[0]

        st.success(f'The predicted cumulative GPA is: {prediction:.2f}')