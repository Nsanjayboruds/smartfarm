import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="SmartFarm",
    page_icon="🌾",
    layout="centered"
)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('Crop_recommendation.csv')
    X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    return X, y, df

@st.cache_resource
def load_model():
    try:
        return pickle.load(open('RF.pkl', 'rb'))
    except:
        # Train and save model if not found
        X, y, _ = load_data()
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
        RF = RandomForestClassifier(n_estimators=20, random_state=5)
        RF.fit(Xtrain, Ytrain)
        pickle.dump(RF, open('RF.pkl', 'wb'))
        return RF

# Load resources
X, y, df = load_data()
RF_Model_pkl = load_model()
img = Image.open("crop.png")

# App functions
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

def show_crop_image(crop_name):
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.error("Image not found for the predicted crop.")

# Main app
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🌾 SmartFarm: Smart Crop Recommendations</h1>", unsafe_allow_html=True)
    st.image(img, width=200)
    
    st.sidebar.markdown("<h2 style='color: green;'>🌱 SmartFarm</h2>", unsafe_allow_html=True)
    st.sidebar.header("Enter Crop Details")

    # Inputs
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

    if st.sidebar.button("Predict"):
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            st.success(f"The recommended crop is: {prediction[0]}")
            show_crop_image(prediction[0])

if __name__ == '__main__':
    main()