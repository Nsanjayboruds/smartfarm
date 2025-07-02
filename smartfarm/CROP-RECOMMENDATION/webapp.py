import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ✅ Page Config
st.set_page_config(page_title="SmartFarm", page_icon="🌾", layout="centered")

# ✅ Header Image
img_path = os.path.join(os.path.dirname(__file__), "crop.png")
if os.path.exists(img_path):
    st.image(Image.open(img_path), use_container_width=True)
else:
    st.warning("Image 'crop.png' not found.")

# ✅ Dataset Load
csv_path = os.path.join(os.path.dirname(__file__), 'Crop_recommendation.csv')
if not os.path.exists(csv_path):
    st.error("❌ 'Crop_recommendation.csv' not found.")
    st.stop()

df = pd.read_csv(csv_path)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Load or Train Model
model_path = os.path.join(os.path.dirname(__file__), 'RF.pkl')
if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    with open(model_path, 'rb') as f:
        RF_Model = pickle.load(f)
else:
    RF_Model = RandomForestClassifier(n_estimators=20, random_state=5)
    RF_Model.fit(Xtrain, Ytrain)
    with open(model_path, 'wb') as f:
        pickle.dump(RF_Model, f)

# ✅ Prediction Logic
def predict_crop(n, p, k, temp, hum, ph_val, rain):
    data = np.array([[n, p, k, temp, hum, ph_val, rain]])
    return RF_Model.predict(data)[0]

# ✅ Optional Crop Image Display
def show_crop_image(crop):
    img_file = os.path.join("crop_images", crop.lower() + ".jpg")
    if os.path.exists(img_file):
        st.image(img_file, caption=f"Recommended Crop: {crop}", use_container_width=True)
    else:
        st.info(f"No image found for '{crop}'.")

# ✅ Main App
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🌱 SmartFarm: Crop Recommendations</h1>", unsafe_allow_html=True)
    st.sidebar.header("🌾 Enter Soil & Climate Info")

    n = st.sidebar.number_input("Nitrogen (N)", 0.0, 140.0, 0.0, 0.5)
    p = st.sidebar.number_input("Phosphorus (P)", 0.0, 145.0, 0.0, 0.5)
    k = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, 0.0, 0.5)
    temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
    hum = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 50.0, 0.5)
    ph_val = st.sidebar.number_input("pH Level", 0.0, 14.0, 6.5, 0.1)
    rain = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, 0.5)

    if st.sidebar.button("🌾 Recommend Crop"):
        prediction = predict_crop(n, p, k, temp, hum, ph_val, rain)
        st.success(f"✅ Recommended Crop: **{prediction.upper()}**")
        show_crop_image(prediction)

    with st.expander("📊 Show Sample Data"):
        st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
