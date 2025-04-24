import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
with open("model/classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("model/regressor.pkl", "rb") as f:
    regressor = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Load association rules safely
rules_df = pd.read_csv("model/rules.csv")
rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))
rules_df["consequents"] = rules_df["consequents"].apply(lambda x: list(eval(x)) if isinstance(x, str) else list(x))

st.title("Diabetes Prediction App")
st.write("This app uses machine learning to predict your BMI, diabetes risk, cluster profile, and health behavior insights.")

st.header("Enter Your Health Details")

def to_binary(val):
    return 1 if val == "Yes" else 0

# Inputs
high_bp = st.selectbox("Do you have high blood pressure?", ["Yes", "No"])
high_chol = st.selectbox("Do you have high cholesterol?", ["Yes", "No"])
chol_check = st.selectbox("Have you done a cholesterol check?", ["Yes", "No"])
smoker = st.selectbox("Do you smoke?", ["Yes", "No"])
stroke = st.selectbox("Have you had a stroke?", ["Yes", "No"])
heart_disease = st.selectbox("Heart disease or heart attack?", ["Yes", "No"])
phys_activity = st.selectbox("Do you engage in physical activity?", ["Yes", "No"])
diff_walk = st.selectbox("Do you have difficulty walking?", ["Yes", "No"])
genhlth = st.slider("General health (1 = Excellent, 5 = Poor)", 1, 5, 3)
menthlth = st.slider("Days mental health not good (last 30 days)", 0, 30, 5)
physhlth = st.slider("Days physical health not good (last 30 days)", 0, 30, 5)
sex = st.selectbox("Biological sex", ["Male", "Female"])
age = st.slider("Age group (1 = 18–24, ..., 13 = 80+)", 1, 13, 5)
sex_bin = 1 if sex == "Male" else 0

# Input for regression
reg_input_dict = {
    "HighBP": to_binary(high_bp),
    "HighChol": to_binary(high_chol),
    "CholCheck": to_binary(chol_check),
    "Smoker": to_binary(smoker),
    "Stroke": to_binary(stroke),
    "HeartDiseaseorAttack": to_binary(heart_disease),
    "PhysActivity": to_binary(phys_activity),
    "GenHlth": genhlth,
    "MentHlth": menthlth,
    "PhysHlth": physhlth,
    "DiffWalk": to_binary(diff_walk),
    "Sex": sex_bin,
    "Age": age
}
reg_input_df = pd.DataFrame([reg_input_dict])

# Predict BMI
bmi_pred = regressor.predict(reg_input_df)[0]

# Input for classifier
cls_input_dict = reg_input_dict.copy()
cls_input_dict["BMI"] = bmi_pred
cls_feature_order = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
    'Stroke', 'HeartDiseaseorAttack', 'PhysActivity',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
    'Sex', 'Age'
]
cls_input_data = [cls_input_dict.get(col, 0) for col in cls_feature_order]
cls_input_df = pd.DataFrame([cls_input_data], columns=cls_feature_order).astype(np.float64)

# Debug Info
if st.checkbox("Show debug info"):
    st.write("Classifier Input DataFrame:")
    st.dataframe(cls_input_df)

# Prediction
if st.button("Predict Diabetes Status"):
    try:
        pred = classifier.predict(cls_input_df)[0]
        probs = classifier.predict_proba(cls_input_df)[0]
        classes = {0: "No Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}

        st.subheader("Diabetes Prediction")
        st.success(f"You are predicted to have: **{classes[pred]}**")

        st.write("### Probability Breakdown")
        for i, p in enumerate(probs):
            st.write(f"- {classes[i]}: {p*100:.2f}%")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# BMI Output
st.header("Predicted BMI")
st.write(f"Estimated BMI: **{bmi_pred:.2f}**")

# Clustering
cluster_input = pd.DataFrame([{
    "BMI": bmi_pred,
    "Age": age,
    "GenHlth": genhlth,
    "PhysHlth": physhlth,
    "MentHlth": menthlth
}])
scaled_input = scaler.transform(cluster_input)
cluster_label = kmeans.predict(scaled_input)[0]
cluster_meaning = {
    0: "Low-risk, healthy lifestyle",
    1: "Moderate risk, needs attention",
    2: "High BMI and physical health issues",
    3: "Older age, higher mental stress"
}

st.subheader("🧩 Cluster Analysis")
st.info(f"You're grouped in **Cluster {cluster_label}**: {cluster_meaning.get(cluster_label)}")

# Association Rule Matching
user_items = [
    f"HighBP_Yes" if to_binary(high_bp) else "HighBP_No",
    f"HighChol_Yes" if to_binary(high_chol) else "HighChol_No",
    f"Smoker_Yes" if to_binary(smoker) else "Smoker_No",
    f"PhysActivity_Yes" if to_binary(phys_activity) else "PhysActivity_No"
]
matched = rules_df[rules_df['antecedents'].apply(lambda ant: all(x in user_items for x in ant))]
if not matched.empty:
    rule = matched.iloc[0]
    st.subheader("📊 Health Pattern Insight")
    st.warning(
        f"If: **{', '.join(rule['antecedents'])}** → Then: **{', '.join(rule['consequents'])}**\n\n"
        f"Confidence: {rule['confidence']*100:.2f}%, Lift: {rule['lift']:.2f}"
    )
