# ✈️ Flight Delay Prediction (Minutes Early/Delayed)

This project predicts the **arrival delay in minutes** for upcoming flights.  
Delays are positive values, early arrivals are negative values.

## 🚀 Features
- Data preprocessing with **tidymodels**
- Machine learning model: **XGBoost (regression)**
- Automatic feature engineering (departure time, day of week, flight distance…)
- Handles categorical variables (airline, origin, destination, aircraft type)
- Model saved for future inference
- Prediction script for **new upcoming flights**

## 📂 Project Structure

flight-delay-prediction/
├── data/
│ ├── flight_delays.csv
│ ├── upcoming_flights.csv
│ └── predictions.csv
├── models/
│ └── arrival_delay_xgb_model.rds
├── scripts/
│ ├── train_model.R
│ └── predict_new_flights.R
└── README.md


## 📊 Dataset
The dataset used in this project comes from Kaggle:  
👉 [Flight Delays Dataset on Kaggle](https://www.kaggle.com/datasets/umeradnaan/flight-delays-dataset)

You can download it and place `flight_delays.csv` inside the `data/` folder.

## 🛠️ Installation
```bash
# Install dependencies in R
install.packages(c("tidyverse", "lubridate", "tidymodels", "vip"))
