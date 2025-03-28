# 🎢 Disney World Wait Time Predictor

This Streamlit application helps visualize and predict wait times for attractions at **Disney World**.  
It allows users to explore historical posted wait time data, simulate busyness levels throughout the year, and generate optimized daily itineraries based on preferences.

---

## 📌 Features

### 📊 Data Dashboard

Visualize average wait times by weekday and holiday using:

- Heatmaps  
- Bar charts  
- Line charts  

**Filters:**

- Date range  
- Attraction  
- Weekday  
- Holiday status  

**Analysis Options:**

- Analyze ride-specific trends over time  
- Compare attractions to find those with the lowest wait times  

---

### 🔎 Recommendation System

- Suggest attractions with low average wait times  
- Based on user-defined thresholds  

---

### 📆 Calendar Simulation

- Predict wait times for each day of a selected month  
- Calendar view color-coded by crowd levels (e.g. “Busy”, “Very Low”)  
- Analyze wait time trends by hour  
- Compare selected attraction with others  

---

### 🧠 Intelligent Itinerary Generator (Experimental)

- Interpolates data for each day in 30-minute intervals  
- Users choose a date and arrival time  
- Generates a ride plan using a **greedy algorithm** (shortest wait times first)  
- Customizable:
  - Number of rides  
  - Ride and walk durations  

---

## 🚀 Getting Started

**1. Clone the Repository:**

git clone https://github.com/gulnisunay/disney-waiting-times.git
cd disney-waiting-times

## 2. Set Up the Environment

conda env create -f environment.yml
conda activate disney_env_v2

## 3. Run the App

streamlit run Disney_streamlit_app.py

---

### 🧪 Optional: Model Training

To train your own wait time prediction model:
python train_predict_model.py

This will generate:

wait_time_model.pkl

model_features.pkl

These can be used in the app for machine learning-based predictions.