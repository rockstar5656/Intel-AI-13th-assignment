import streamlit as st
import pickle
import numpy as np
import os

@st.cache_resource
def load_model():
    # Get the correct path for Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), 'models/weather_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']

st.title('🌦️ Weather Rainfall Prediction')
st.write('Predict rainfall based on weather parameters')

# Input widgets
col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input('Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0)
with col2:
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=60.0)
with col3:
    wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=100.0, value=10.0)

# Prediction button
if st.button('Predict Rainfall'):
    input_features = np.array([[temp, humidity, wind_speed]])
    scaled_input = scaler.transform(input_features)
    prediction = model.predict(scaled_input)
    
    st.subheader('Prediction Result')
    st.metric(label="Expected Rainfall", value=f"{prediction[0]:.2f} mm")