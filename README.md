# 🏠 Costa Rica Household Poverty Prediction System

## 📌 Project Overview
This project aims to predict the poverty level of households in Costa Rica using socio-economic indicators. It features a complete Machine Learning pipeline—from advanced feature engineering and data balancing to an optimized **XGBoost** classification model, all wrapped in an interactive **Streamlit** web application.

## 🧪 How to Test the Application (Important!)
To easily evaluate the model's performance and test the Streamlit UI, **please use the specific datasets provided in this repository**:

1. **`test_scaled.csv`**: Upload this file to the UI to see how the AI predicts completely unseen data. 
2. **`train_scaled.csv`**: You can also use this file to test the model on the data distribution it was trained on.

💡 *Note: These specific files have already passed through our custom Feature Engineering and Scaling pipeline, meaning they are perfectly formatted and ready for the model to process instantly without any feature mismatch errors.*

## 🚀 Installation & Running the App

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/elsaeedmohamed/Depi-project-.git](https://github.com/elsaeedmohamed/Depi-project-.git)
   cd Depi-project-
   Install the required dependencies:
Make sure you have Python installed, then run:

Bash
pip install pandas xgboost scikit-learn streamlit joblib matplotlib seaborn openpyxl
Run the Streamlit Dashboard:

Bash
streamlit run app.py
The application will automatically open in your default web browser.

🧠 Key Features & Methodologies
Advanced Feature Engineering: Extracted meaningful socio-economic insights by creating custom mathematical ratios (e.g., children_ratio, rooms_per_person, educ_overcrowding).

Handling Imbalanced Data: Addressed the massive class imbalance using oversampling techniques to ensure fair predictions for vulnerable minority classes.

Algorithm Selection: Utilized XGBoost for its superior performance on tabular data and complex non-linear boundaries.

Interactive Deployment: Built a user-friendly UI using Streamlit that allows users to upload raw CSV files, processes them under the hood, and returns visualized poverty predictions and downloadable Excel/CSV reports.

🛠️ Built With
Python 🐍

XGBoost

Scikit-Learn

Pandas & NumPy

Streamlit

Matplotlib & Seaborn

Developed with clean code and passion. 💻✨
