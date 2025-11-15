# End to End Power Consumption Prediction Project
# 🔌 Multi-Zone Electric Power Consumption Forecasting (Machine Learning + Streamlit)

This project predicts **electric power consumption across three different zones** using weather data and engineered timestamp features.  
It includes a complete ML pipeline, EDA, model evaluation, and a fully deployable **Streamlit web application**.

---

# 📌 Problem Statement

Electricity consumption varies drastically due to weather, time, season, and human activity.  
Unpredictable demand can lead to:

- Grid instability  
- High operational cost  
- Power shortages / brownouts  
- Inefficient resource planning  

Accurate forecasting is essential to maintain **grid stability**, **reduce costs**, and **optimize supply**.

---

# 🎯 Goal of the Project

To build a **high-accuracy predictive model** capable of forecasting **Zone 1, Zone 2, and Zone 3** power consumption using:

- Weather features  
- Time-based features  
- Multi-output regression  

---

# 🛠️ Tools and Technologies Used

- **Python**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Scikit-Learn**
- **XGBoost**
- **Streamlit**
- **Joblib**
- **Jupyter Notebook / VS Code**

---

# 📊 Methodology (Project Workflow)

### **1. Data Preparation**
- Loaded dataset  
- Handled missing/duplicate values  
- Extracted temporal features:  
  - Month  
  - Day  
  - Hour  
  - Minute  
  - DayOfWeek  
  - IsWeekend  
- (Earlier scaling was used, but final model uses **raw features**)  

---

### **2. Exploratory Data Analysis (EDA)**
- Time-series visualization of power consumption  
- Monthly & hourly consumption pattern analysis  
- Weather vs Power analysis (Temperature/Humidity correlation)  
- Correlation heatmap  

This helped identify **seasonality, trends, and non-linear patterns**.

---

### **3. Machine Learning Modeling**
- 80/20 Train-test split  
- Trained 8 models:  
  - Linear Regression  
  - KNN  
  - Decision Tree  
  - Random Forest  
  - XGBoost   
- Used **MultiOutputRegressor** to predict all 3 zones simultaneously  
- Evaluated using:  
  - **R² Score**  
  - **Mean Absolute Error (MAE)**  

---

# 🧠 Final Selected Model

### ⭐ **Random Forest Regressor (Best Model)**  
- R² Score: **> 0.99**  
- Lowest MAE: **~372**  
- Captures highly non-linear relationships  
- Very stable and consistent  

---

# 🖥️ Streamlit App

The project includes a deployed Streamlit web app that:

- Accepts raw weather inputs  
- Extracts temporal features automatically  
- Predicts **Zone 1**, **Zone 2**, **Zone 3** consumption  
- Uses the trained ML model stored as `model.pkl`

---


<img width="1908" height="962" alt="Screenshot 2025-11-15 130752" src="https://github.com/user-attachments/assets/0f84d06b-dc45-4102-843f-be6eff5867b6" />
<img width="1916" height="924" alt="Screenshot 2025-11-15 130836" src="https://github.com/user-attachments/assets/f6df09b1-3a0f-4745-899f-e9c0b60e4473" />
