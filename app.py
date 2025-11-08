import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="ECG Classifier", layout="wide")

@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('model_lite')
        return model
    except (IOError, ImportError):
        st.error("Error loading model: The 'model_lite' folder is missing or corrupt.")
        st.stop()
        
model = load_my_model()

# --- 2. SIDEBAR INFO ---
st.sidebar.header("Project Resources")
st.sidebar.markdown("[Data Source (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)")
st.sidebar.markdown("[View on GitHub](https://github.com/Mohit9912/ECG-arrhythmia-classifier)")

# --- 3. MAIN PAGE ---
st.title("🩺 ECG Arrhythmia Classifier")
st.write("**Upload a patient's ECG data file (CSV) for a complete automated diagnosis.**")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, header=None)
        
        # Validate file
        if df.shape[1] != 188:
            st.error(f"Error: File has {df.shape[1]} columns, expected 188.")
            st.stop()

        # Prepare data (split signals from labels)
        X_data = df.drop(187, axis=1).values  # Convert to numpy array immediately
        y_true = df[187].astype(int).values

        # Reshape for the model (samples, 187, 1)
        X_data_reshaped = X_data.reshape(len(X_data), 187, 1)

        st.success(f"File loaded: {len(df)} heartbeats detected.")
        st.markdown("---")

        # ========================================================
        # NEW SECTION: WHOLE FILE AUTOMATED DIAGNOSIS
        # ========================================================
        st.header("📋 Automated Diagnostic Report")

        # 1. Run prediction on the ENTIRE file at once (Batch Prediction)
        with st.spinner("Analyzing all heartbeats... please wait..."):
            # This line predicts ALL 4000+ rows in one go
            all_predictions = model.predict(X_data_reshaped, verbose=0)
            
            # Convert probabilities to 0 (Normal) or 1 (Abnormal)
            predicted_classes = (all_predictions > 0.5).astype(int).flatten()

        # 2. Calculate Statistics
        total_beats = len(predicted_classes)
        abnormal_beats = np.sum(predicted_classes == 1)
        normal_beats = np.sum(predicted_classes == 0)
        arrhythmia_percentage = (abnormal_beats / total_beats) * 100

        # 3. Display Metrics visually
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Heartbeats", total_beats)
        m2.metric("Normal Beats Found", int(normal_beats), delta="Healthy")
        m3.metric("Abnormal Beats Found", int(abnormal_beats), 
                  delta="-Risk" if abnormal_beats > 0 else "None",
                  delta_color="inverse")

        # 4. FINAL DIAGNOSIS LOGIC
        # If more than 10% of beats are abnormal, flag the patient.
        st.subheader("Final Diagnosis:")
        if arrhythmia_percentage > 10:
             st.error(f"🚨 **ARRHYTHMIA DETECTED**")
             st.write(f"This patient's recording shows **{arrhythmia_percentage:.1f}%** abnormal heartbeats.")
        else:
             st.success(f"✅ **NORMAL SINUS RHYTHM**")
             st.write(f"This patient's recording is mostly healthy ({arrhythmia_percentage:.1f}% abnormal detected, likely noise).")

        st.markdown("---")
        # ========================================================
        # END NEW SECTION
        # ========================================================

        # (Optional) Detailed Drill-Down
        with st.expander("🔬 Drill Down: Inspect Individual Heartbeats"):
            beat_index = st.slider("Select a heartbeat to view:", 0, len(df) - 1, 0)
            selected_beat = X_data[beat_index]
            
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(selected_beat)
            ax.set_title(f"Beat #{beat_index}")
            ax.set_yticks([])
            st.pyplot(fig)
            
            # Show prediction for just this one beat
            pred = predicted_classes[beat_index]
            truth = y_true[beat_index]
            st.write(f"**Model Prediction:** {'Abnormal' if pred == 1 else 'Normal'}")
            st.write(f"**Actual Label:** {'Abnormal' if truth == 1 else 'Normal'}")

    except Exception as e:
        st.error(f"Error processing file: {e}")