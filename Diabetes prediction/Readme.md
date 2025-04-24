Diabetes Prediction and Health Profiling App:

An interactive machine learning web application built with Streamlit that predicts diabetes risk based on user-input health indicators. The app also performs clustering analysis to group users into health risk categories for better profiling and insight.

Features:

Diabetes Risk Prediction using a classification model (90%+ accuracy).

Health Profiling with K-Means clustering based on user health features.

Feature Selection for model optimization using correlation and feature importance.

Clean and intuitive Streamlit UI for real-time input and predictions.

Tech Stack:

Python, Pandas, NumPy

Scikit-learn (Classification, Clustering, Feature Selection)

Streamlit (Web UI)

Matplotlib, Seaborn (for visualizations if included)

📊 Dataset:

This project utilizes the Diabetes Health Indicators Dataset from Kaggle, which includes 253,680 survey responses from the cleaned BRFSS 2015 dataset. The dataset is balanced with a 50-50 split between respondents with no diabetes and those with either prediabetes or diabetes. The target variable Diabetes_binary has two classes.
Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

🚀 How to Run:

Clone the repository:
git clone https://github.com/your-username/diabetes-prediction-app.git
cd diabetes-prediction-app

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

