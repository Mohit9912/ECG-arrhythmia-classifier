import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & MODEL LOADING ---

# Set the page configuration for a cleaner, wider layout
st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")

# Use st.cache_resource to load the model only once (for efficiency)
@st.cache_resource
def load_my_model():
    """
    Loads the trained Keras model from the 'model_lite' folder.
    The @st.cache_resource decorator ensures this function only runs once,
    making the app load instantly on subsequent runs.
    """
    try:
        model = tf.keras.models.load_model('/home/mohit-k-d-bme/Desktop/ECG-arrhythmia-classifier/model_lite')
        return model
    except (IOError, ImportError) as e:
        # Stop the app if the model (the "brain") is missing
        st.error("Error loading model: The 'model_lite' folder is missing or corrupt.")
        st.stop()
        
model = load_my_model()

# --- 2. SIDEBAR & APP INTRODUCTION ---

st.sidebar.title("About This Project")
st.sidebar.info(
    "**Project: ECG Arrhythmia Classifier**\n\n"
    "This app is a 7th-semester BME project demonstrating end-to-end "
    "Machine Learning deployment.\n\n"
    "It uses a lightweight 1D-CNN (Convolutional Neural Network) "
    "trained on the PTBDB dataset to classify heartbeats as **Normal** or **Abnormal**."
)
# TODO: Update this link to your "big data" GitHub project
st.sidebar.markdown(
    "Want to see the full, 5-class analysis on the 600MB dataset? \n"
    "Check out the [full research notebook here](https://github.com/your-username/your-MAIN-repo-link)."
)

st.title("🩺 ECG Arrhythmia Classifier")
st.write(
    "**Welcome!** This app uses a deep learning model to classify heartbeats. "
    "Upload a CSV file from the [PTBDB dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) "
    "(like `ptbdb_normal.csv` or `ptbdb_abnormal.csv`) to get started."
)

# --- 3. THE INTERACTIVE FILE UPLOADER ---

uploaded_file = st.file_uploader("Choose a CSV file to analyze", type="csv")

if uploaded_file is None:
    # This is the "home page" state, before a file is uploaded
    st.info("Please upload a CSV file to begin analysis.")

else:
    # --- 4. FILE PROCESSING & ROBUSTNESS CHECKS ---
    
    # This block runs *after* a user uploads a file
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file, header=None)
        
        # --- Robustness Check 1: Is the file empty? ---
        if df.empty:
            st.error("Error: The uploaded file is empty.")
        
        # --- Robustness Check 2: Does it have the correct shape? ---
        # The model was trained on 187 signal points. The file has 188 columns (187 + 1 label).
        elif df.shape[1] != 188:
            st.error(f"Error: Expected 188 columns, but got {df.shape[1]}.")
            st.write("Please upload a file from the 'ptbdb_normal.csv' or 'ptbdb_abnormal.csv' dataset.")
        
        # --- 5. THE MAIN APP INTERFACE (FILE UPLOADED) ---
        else:
            st.success(f"File '{uploaded_file.name}' uploaded successfully! Found {len(df)} heartbeats.")
            st.markdown("---")

            # Separate signals (X) and true labels (y)
            X_data = df.drop(187, axis=1)  # All columns *except* the last one
            y_true = df[187].astype(int)   # The last column (the "true" answer)
            
            # --- Interactivity: Add a slider ---
            beat_index = st.slider("Select a heartbeat to analyze:", 0, len(X_data) - 1, 0)
            
            # Get the single beat the user selected with the slider
            selected_beat = X_data.iloc[beat_index]
            true_label = y_true.iloc[beat_index]
            true_label_name = "Abnormal" if true_label == 1 else "Normal"

            # --- 6. PLOT, PREDICT, & DISPLAY ---
            
            # Create two columns for a clean layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # --- Plot the selected heartbeat ---
                st.subheader(f"Heartbeat #{beat_index} (True Label: {true_label_name})")
                
                fig, ax = plt.subplots()
                ax.plot(selected_beat)
                ax.set_title("Selected Heartbeat Signal")
                ax.set_xlabel("Time (data points)")
                ax.set_ylabel("Signal Amplitude")
                st.pyplot(fig)

            with col2:
                # --- Run the classification ---
                st.subheader("Model's Classification")
                
                # 1. Pre-process the selected beat for the model
                # Reshape to (1, 187, 1) as required by the model
                data_for_model = np.array(selected_beat).reshape(1, 187, 1)
                
                # 2. Make prediction
                prediction_prob = model.predict(data_for_model)[0][0]
                prediction_class = 1 if prediction_prob > 0.5 else 0
                prediction_name = "Abnormal" if prediction_class == 1 else "Normal"
                
                # 3. Show the result using st.metric for a clean look
                is_correct = (prediction_class == true_label)
                
                if prediction_class == 1:
                    st.error(f"**Prediction: {prediction_name}**", icon="🚨")
                else:
                    st.success(f"**Prediction: {prediction_name}**", icon="✅")

                # Show a "Correct" or "Incorrect" delta
                st.metric(
                    label="Prediction Accuracy",
                    value=f"{prediction_prob*100:.1f}%" if prediction_class==1 else f"{(1-prediction_prob)*100:.1f}%",
                    delta="Prediction: Correct" if is_correct else "Prediction: Incorrect",
                    delta_color="normal" if is_correct else "inverse"
                )
                
                # --- Show detailed analysis in an expander ---
                with st.expander("Show Analysis Details"):
                    st.write(f"**True Label:** `{true_label_name}` (Class {true_label})")
                    st.write(f"**Model Prediction:** `{prediction_name}` (Class {prediction_class})")
                    st.write(f"**Model's Raw Probability (Output):** `{prediction_prob:.4f}`")
                    st.info("The model uses a 0.5 threshold. A value > 0.5 is classified as 'Abnormal' (1), and <= 0.5 is 'Normal' (0).")

    except Exception as e:
        # Catch any other potential errors (e.g., corrupt file)
        st.error(f"An error occurred while processing the file: {e}")