import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multi Disease Predictor", layout="centered")

st.title("🩺 Multi Disease Prediction System")
st.markdown(
    "Predict the likelihood of Kidney, Liver, or Parkinsons disease using trained models. "
    "Choose a disease from the sidebar, enter patient data, and see the probability and confidence visualization."
)

# Sidebar
disease = st.sidebar.selectbox(
    "Select Disease",
    ["Kidney Disease", "Liver Disease", "Parkinsons"]
)

base_path = os.path.dirname(__file__)

sidebar_model_status = st.sidebar.container()
if disease == "Kidney Disease":
    model_file = 'kidney_model.pkl'
    scaler_file = 'scaler.pkl'
elif disease == "Liver Disease":
    model_file = 'liver_model.pkl'
    scaler_file = 'scaler_liver.pkl'
else:
    model_file = 'parkinsons_model.pkl'
    scaler_file = 'scaler_parkinsons.pkl'

missing_sidebar_files = [file for file in [model_file, scaler_file] if not os.path.exists(os.path.join(base_path, file))]
sidebar_model_status.markdown("## Model Status")
if missing_sidebar_files:
    sidebar_model_status.warning(
        f"Missing files for {disease}: {', '.join(missing_sidebar_files)}"
    )
    sidebar_model_status.info("Install or train the missing model files to enable this disease prediction.")
else:
    sidebar_model_status.success(f"{disease} model ready")


def load_model_scaler(model_file, scaler_file):
    model_path = os.path.join(base_path, model_file)
    scaler_path = os.path.join(base_path, scaler_file)

    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(model_file)
    if not os.path.exists(scaler_path):
        missing_files.append(scaler_file)

    if missing_files:
        st.warning(
            f"Missing model files: {', '.join(missing_files)}. "
            "To enable this prediction, place these files in the app folder or train the model using the training scripts."
        )
        if disease != "Kidney Disease":
            st.info(
                "Liver and Parkinsons predictions will work once the missing files are added. "
                "For now, Kidney Disease is available if its model files are present."
            )
        return None, None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def load_feature_columns(file_name):
    path = os.path.join(base_path, file_name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ================= KIDNEY =================
if disease == "Kidney Disease":
    model, scaler = load_model_scaler('kidney_model.pkl', 'scaler.pkl')
    feature_columns = load_feature_columns('feature_columns.pkl')
    st.header("Kidney Disease Prediction")
    st.markdown("Enter kidney test values below and click Predict to see the disease probability.")

    with st.form(key="kidney_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50, key="kid_age")
            bp = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0, format="%.1f", key="kid_bp")
            sg = st.number_input("Specific Gravity", min_value=1.0, max_value=2.0, value=1.02, format="%.3f", key="kid_sg")
            al = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=1.0, format="%.1f", key="kid_al")
            su = st.number_input("Sugar", min_value=0.0, max_value=10.0, value=0.0, format="%.1f", key="kid_su")
            bgr = st.number_input("Blood Glucose Random", min_value=0.0, max_value=500.0, value=100.0, format="%.1f", key="kid_bgr")
            bu = st.number_input("Blood Urea", min_value=0.0, max_value=500.0, value=20.0, format="%.1f", key="kid_bu")
            sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0, format="%.2f", key="kid_sc")
        with col2:
            sod = st.number_input("Sodium", min_value=0.0, max_value=200.0, value=135.0, format="%.1f", key="kid_sod")
            pot = st.number_input("Potassium", min_value=0.0, max_value=20.0, value=4.5, format="%.2f", key="kid_pot")
            hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=12.0, format="%.2f", key="kid_hemo")
            pcv = st.number_input("Packed Cell Volume", min_value=0.0, max_value=60.0, value=40.0, format="%.1f", key="kid_pcv")
            wc = st.number_input("White Blood Cell Count", min_value=0.0, max_value=20000.0, value=8000.0, format="%.0f", key="kid_wc")
            rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=6.0, value=4.5, format="%.2f", key="kid_rc")
        col3, col4 = st.columns(2)
        with col3:
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], key="kid_rbc")
            pc = st.selectbox("Pus Cell", ["normal", "abnormal"], key="kid_pc")
            pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], key="kid_pcc")
            ba = st.selectbox("Bacteria", ["notpresent", "present"], key="kid_ba")
            htn = st.selectbox("Hypertension", ["no", "yes"], key="kid_htn")
        with col4:
            dm = st.selectbox("Diabetes Mellitus", ["no", "yes"], key="kid_dm")
            cad = st.selectbox("Coronary Artery Disease", ["no", "yes"], key="kid_cad")
            appet = st.selectbox("Appetite", ["good", "poor"], key="kid_appet")
            pe = st.selectbox("Pedal Edema", ["no", "yes"], key="kid_pe")
            ane = st.selectbox("Anemia", ["no", "yes"], key="kid_ane")
        submitted = st.form_submit_button("Predict Kidney Disease")

    if submitted and model is not None and scaler is not None and feature_columns is not None:
        row = {col: 0 for col in feature_columns}
        row.update({
            'age': age,
            'bp': bp,
            'sg': sg,
            'al': al,
            'su': su,
            'bgr': bgr,
            'bu': bu,
            'sc': sc,
            'sod': sod,
            'pot': pot,
            'hemo': hemo,
            'pcv': pcv,
            'wc': wc,
            'rc': rc,
            'rbc_normal': 1 if rbc == 'normal' else 0,
            'pc_normal': 1 if pc == 'normal' else 0,
            'pcc_present': 1 if pcc == 'present' else 0,
            'ba_present': 1 if ba == 'present' else 0,
            'htn_yes': 1 if htn == 'yes' else 0,
            'dm_yes': 1 if dm == 'yes' else 0,
            'cad_yes': 1 if cad == 'yes' else 0,
            'appet_poor': 1 if appet == 'poor' else 0,
            'pe_yes': 1 if pe == 'yes' else 0,
            'ane_yes': 1 if ane == 'yes' else 0,
        })
        data = pd.DataFrame([row], columns=feature_columns)
        data = scaler.transform(data)
        result = model.predict(data)
        prob = model.predict_proba(data)

        if result[0] == 1:
            st.error(f"High Risk of Kidney Disease (Confidence: {prob[0][1]:.2f})")
        else:
            st.success(f"Low Risk of Kidney Disease (Confidence: {prob[0][0]:.2f})")

        st.write(f"**Probability:** Low risk {prob[0][0]:.2f}, High risk {prob[0][1]:.2f}")
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], prob[0], color=["#2ecc71", "#e74c3c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

# ================= LIVER =================
elif disease == "Liver Disease":
    model, scaler = load_model_scaler('liver_model.pkl', 'scaler_liver.pkl')
    st.header("Liver Disease Prediction")
    st.markdown("Enter liver test values below to predict the disease probability.")

    with st.form(key="liver_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=50, key="liver_age")
            total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0, value=1.0, format="%.2f", key="liver_tb")
        with col2:
            alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0, max_value=1000.0, value=100.0, format="%.1f", key="liver_ap")
        submitted = st.form_submit_button("Predict Liver Disease")

    if submitted and model is not None and scaler is not None:
        data = np.array([[age, total_bilirubin, alkaline_phosphotase]])
        data = scaler.transform(data)
        result = model.predict(data)
        prob = model.predict_proba(data)[0]

        if result[0] == 1:
            st.error(f"High Risk of Liver Disease (Confidence: {prob[1]:.2f})")
        else:
            st.success(f"Low Risk of Liver Disease (Confidence: {prob[0]:.2f})")

        st.write(f"**Probability:** Low risk {prob[0]:.2f}, High risk {prob[1]:.2f}")
        fig, ax = plt.subplots()
        ax.bar(["Low Risk", "High Risk"], prob, color=["#2ecc71", "#e74c3c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

# ================= PARKINSONS =================
elif disease == "Parkinsons":
    model, scaler = load_model_scaler('parkinsons_model.pkl', 'scaler_parkinsons.pkl')
    st.header("Parkinsons Prediction")
    st.markdown("Enter the vocal measurement values below for prediction.")

    with st.form(key="parkinsons_form"):
        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0, value=120.0, format="%.3f", key="park_fo")
            fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=300.0, value=150.0, format="%.3f", key="park_fhi")
        with col2:
            flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0, value=110.0, format="%.3f", key="park_flo")
        submitted = st.form_submit_button("Predict Parkinsons")

    if submitted and model is not None and scaler is not None:
        data = np.array([[fo, fhi, flo]])
        data = scaler.transform(data)
        result = model.predict(data)
        prob = model.predict_proba(data)[0]

        if result[0] == 1:
            st.error(f"Parkinsons Detected (Confidence: {prob[1]:.2f})")
        else:
            st.success(f"No Parkinsons Detected (Confidence: {prob[0]:.2f})")

        st.write(f"**Probability:** No Parkinsons {prob[0]:.2f}, Parkinsons {prob[1]:.2f}")
        fig, ax = plt.subplots()
        ax.bar(["No Parkinsons", "Parkinsons"], prob, color=["#2ecc71", "#e74c3c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

