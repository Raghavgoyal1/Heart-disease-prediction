import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/knn_heart_model.pkl')
scaler = joblib.load('model/heart_scaler.pkl')
expected_columns = joblib.load('model/heart_columns.pkl')

st.title("Heart Disease Prediction App")
st.markdown("Provide the following information to predict the risk of heart stroke:")

age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])

chestPainType = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
restingECG = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
maxHR = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exerciseAngina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.0, 1.0)
stSlope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    # Map friendly UI labels to the exact category codes the model was
    # trained on (see model/heart_columns.pkl). One category per feature
    # is the "baseline" and was dropped during training (drop_first=True),
    # so it's intentionally left out of raw_input below and stays all-zero.
    sex_code = {"Male": "M", "Female": "F"}[sex]                      # baseline: F
    angina_code = {"Yes": "Y", "No": "N"}[exerciseAngina]              # baseline: N
    ecg_code = {
        "Normal": "Normal",
        "ST-T Wave Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH",                        # baseline: LVH
    }[restingECG]

    raw_input = {
        "Age": age,
        "RestingBP": restingBP,
        "Cholesterol": cholesterol,
        "FastingBS": fastingBS,
        "MaxHR": maxHR,
        "Oldpeak": oldpeak,
        "Sex_" + sex_code: 1,
        "ChestPainType_" + chestPainType: 1,
        "RestingECG_" + ecg_code: 1,
        "ExerciseAngina_" + angina_code: 1,
        "ST_Slope_" + stSlope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("High risk of heart stroke. Please consult a doctor.")
    else:
        st.success("Low risk of heart stroke. Keep maintaining a healthy lifestyle.")

