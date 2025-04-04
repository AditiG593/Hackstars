# Hackstars
Welcome to our Hackathon project! This repository is a complete machine learning solution that **predicts client portfolio values for the next 3 years** and **classifies the recommended investment strategy** based on financial and behavioral data.

---

## 🧠 Project Overview

This project aims to assist financial advisors and institutions by:

- 📈 **Forecasting** future portfolio values (for Year 1, 2, and 3)
- 🧭 **Classifying** clients into suitable investment strategies (Conservative, Balanced, Aggressive)
- 🌍 Simulating **macroeconomic scenario impacts** on client portfolios

## 📊 Dataset Visualization
Before diving into modeling, it's crucial to **understand the dataset**. Here are some **histograms, correlation heatmaps, and feature distributions** to gain insights into data patterns.  

All Features 

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Data%20Visualization.png" width="300" height="200" />

Static Client Data

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Static_Client_Data.png" width="300" height="200" />

Time Series Data

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Time_Series_Data.png" width="300" height="200" />

Macro scenarios

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Macro_scenarios.png" width="300" height="200" />


## 🏷️ Classification Models
### 🚀Multilayer Perceptron (MLP)
This model uses a neural network to classify clients into the most suitable investment strategy based on their financial behavior and goals. It captures complex patterns in data, making the strategy recommendation smarter and more personalized.

### 🚀Logistic Regression
Uses a basic yet effective classification approach that handles class imbalance to fairly predict the recommended investment strategy

### 🚀XGBoost Classification
A powerful model that uses advanced features from both static and time-series client data to classify investment strategies (Aggressive, Balanced, Conservative).
It incorporates lag, trend, and macroeconomic signals to boost prediction accuracy.

<img src="https://raw.githubusercontent.com/AditiG593/Hackstars/main/Assets/Classification_XGBoost.png" width="250" height="150" />


## 📈 Regression Models

### 🔢Linear Regression
Predicts client portfolio values for the next 3 years using a simple linear model. Includes data preprocessing, training, and evaluation using MAE & RMSE.

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Regression_Linear.png" width="350" height="250" />

### 🔢XGBoost Regression
Predicts portfolio values for 3 future years using XGBoost. Adds cool engineered features like savings ratio & risk-volatility interaction. Evaluates performance with MAE, RMSE, and R², and shows feature importance 🌟📈

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Regression_XGBoost.png" width="350" height="250" />


Feature Importance

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Regression_Feature_Importance.png" width="300" height="200" />


### 🔢XGBoost Regression with Different Scenarios
-Predicts for specific client UUID
-Applies macro scenario effects to simulate market changes 📉📈

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Regression_XGBoost_2.png" width="500" height="200" />

<img src="https://github.com/AditiG593/Hackstars/blob/main/Assets/Regression_XGBoost_Scenario.png" width="300" height="200" />




