🩺 ECG Arrhythmia Classifier
---

An interactive web app that uses a 1D Convolutional Neural Network (CNN) to classify ECG heartbeats in real-time. This project demonstrates a complete end-to-end Machine Learning workflow, from data processing and model training to a live, deployed web application.


OUTPUT OF MODEL TRAINING SCRIPT 
------ 
```
 --- 1. Libraries Imported ---
--- 2. Data Loaded Successfully ---
--- 3. Data Labeled, Combined, and Shuffled ---
Training data shape: (11641, 187)
Testing data shape: (2911, 187)
New 3D Training shape: (11641, 187, 1)
--- 4. Data Pre-processing Complete ---
/usr/local/lib/python3.12/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
--- 5. Model Built Successfully ---
Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d (Conv1D)                 │ (None, 182, 32)        │           224 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 182, 32)        │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d (MaxPooling1D)    │ (None, 90, 32)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (Conv1D)               │ (None, 88, 64)         │         6,208 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 88, 64)         │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d_1 (MaxPooling1D)  │ (None, 44, 64)         │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 2816)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │        90,144 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │            33 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 96,993 (378.88 KB)

 Trainable params: 96,801 (378.13 KB)

 Non-trainable params: 192 (768.00 B)

--- 6. Model Compiled ---
--- 7. STARTING MODEL TRAINING (This will be fast!) ---
Epoch 1/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 9s 43ms/step - accuracy: 0.8047 - loss: 0.4211 - val_accuracy: 0.7135 - val_loss: 0.6499
Epoch 2/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - accuracy: 0.9477 - loss: 0.1425 - val_accuracy: 0.7080 - val_loss: 0.5650
Epoch 3/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - accuracy: 0.9731 - loss: 0.0811 - val_accuracy: 0.7176 - val_loss: 0.5102
Epoch 4/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9850 - loss: 0.0475 - val_accuracy: 0.7314 - val_loss: 0.5112
Epoch 5/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9911 - loss: 0.0298 - val_accuracy: 0.8052 - val_loss: 0.3805
Epoch 6/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9938 - loss: 0.0203 - val_accuracy: 0.8083 - val_loss: 0.5496
Epoch 7/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9962 - loss: 0.0149 - val_accuracy: 0.8846 - val_loss: 0.2900
Epoch 8/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9975 - loss: 0.0101 - val_accuracy: 0.9814 - val_loss: 0.0586
Epoch 9/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - accuracy: 0.9980 - loss: 0.0077 - val_accuracy: 0.9811 - val_loss: 0.0529
Epoch 10/10
91/91 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.9998 - loss: 0.0037 - val_accuracy: 0.9873 - val_loss: 0.0410
--- 8. MODEL TRAINING COMPLETE ---
--- 9. Lightweight model 'model_lite.keras' saved! ---
Downloading your model file... Please wait.
```

SCREENSHOTS 
---


🚀 Live Demo
---

You can try the live app here:
---
https://ecg-arrhythmia-classifier-mzu5averkxpbyljimyb3vh.streamlit.app/




📖 About This Project
---

This project is a 7th-semester Biomedical Engineering assignment. The goal was to build an accessible tool that can analyze a cardiovascular signal—an Electrocardiogram (ECG)—and automatically detect abnormalities.

This web app serves as a lightweight, fast, and portable demo. It is powered by a 1D-CNN trained on the PTB Diagnostic ECG Database (PTBDB), which contains two classes:

Class 0: Normal Heartbeat

Class 1: Abnormal Heartbeat (Arrhythmia)

The app allows a user to upload their own data (in the CSV format) and instantly plot and classify any heartbeat from the file.

✨ Features
---

📈 Interactive Plotting: Upload a compatible .csv file and instantly plot any heartbeat using a simple slider.

🤖 Live Classification: Get an instant Normal/Abnormal prediction from the trained TensorFlow/Keras model.

🔬 Detailed Analysis: The app provides the model's raw probability score and compares the prediction against the file's "true" label to show its accuracy.

✅ Robust & Efficient: The model is cached in memory for instant performance, and the app includes checks to validate user-uploaded files.

🛠️ Technology Stack
---

Backend & Model: Python, TensorFlow (Keras)

Web Framework: Streamlit

Data Processing: Pandas, NumPy

Plotting: Matplotlib

Deployment: Streamlit Community Cloud

Version Control: Git & GitHub

📂 Project Structure
---
```
ECG-arrhythmia-classifier/
│
├── .git/
├── .gitignore          <-- Ignores virtual env and other junk
├── app.py              <-- The main Streamlit app script
├── model_lite/         <-- The lightweight, trained Keras model
│   ├── config.json
│   ├── metadata.json
│   └── model.weights.h5
├── README.md           <-- You are here!
└── requirements.txt    <-- The "shopping list" of Python libraries
```


🏃‍♂️ How to Run This App Locally
---
If you want to run this project on your own machine:

Clone the repository:
```
git clone https://github.com/Mohit9912/ECG-arrhythmia-classifier.git
```
```
cd ECG-arrhythmia-classifier
```

Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```


Install the required libraries:
```
pip install -r requirements.txt
```


Run the Streamlit app:
```
streamlit run app.py
```


Your browser will automatically open with the app.

