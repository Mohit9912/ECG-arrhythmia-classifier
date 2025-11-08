# --- 1. IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization
from google.colab import files

print("--- 1. Libraries Imported ---")

# --- 2. LOADing DATA ---
# Load both normal and abnormal datasets
try:
    df_normal = pd.read_csv('/content/drive/MyDrive/ECG_arrhythmia_classifier /data /ptbdb_normal.csv', header=None)
    df_abnormal = pd.read_csv('/content/drive/MyDrive/ECG_arrhythmia_classifier /data /ptbdb_abnormal.csv', header=None)
    print("--- 2. Data Loaded Successfully ---")
except FileNotFoundError:
    print("!!! ERROR: Make sure ptbdb_normal.csv and ptbdb_abnormal.csv are uploaded. !!!")
    raise

# --- 3. LABEL DATA (0 = Normal, 1 = Abnormal) ---
# We will use the last column (187) to store our label
df_normal[187] = 0
df_abnormal[187] = 1

# --- 4. COMBINE & SHUFFLE DATA ---
# Stack them on top of each other
df_combined = pd.concat([df_normal, df_abnormal])

# Shuffle the combined dataset. This is a CRITICAL step.
# random_state=42 ensures you get the same shuffle every time
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- 3. Data Labeled, Combined, and Shuffled ---")

# --- 5. CREATE X (signals) and y (labels) ---
# The last column (187) is our label
y = df_combined[187]

# All other columns (0-186) are the signal
X = df_combined.drop(187, axis=1)

# --- 6. SPLIT INTO TRAIN & TEST ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 7. RESHAPE DATA FOR 1D-CNN ---
# The CNN needs data in a 3D shape: [samples, timesteps, features]
# We have [samples, 187]. We add a '1' at the end for "features".
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

print(f"New 3D Training shape: {X_train_cnn.shape}")
print("--- 4. Data Pre-processing Complete ---")

# --- 8. BUILDING THE MODEL ---
# This is a *simpler* model to keep the file size small
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=6, activation='relu', input_shape=(187, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(32, activation='relu'))

# The final layer. 'sigmoid' is used for 2-class (binary) problems.
model.add(Dense(1, activation='sigmoid')) 

print("--- 5. Model Built Successfully ---")
model.summary()

# --- 9. COMPILE THE MODEL ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Use 'binary_crossentropy' for a 2-class problem
    metrics=['accuracy']
)
print("--- 6. Model Compiled ---")

# --- 10. TRAIN THE MODEL ---
print("--- 7. STARTING MODEL TRAINING (This will be fast!) ---")
history = model.fit(
    X_train_cnn, y_train,
    epochs=10, # 10 epochs is enough for this simple problem
    batch_size=128,
    validation_data=(X_test_cnn, y_test)
)
print("--- 8. MODEL TRAINING COMPLETE ---")

# --- 11. SAVE THE FINAL, LIGHTWEIGHT MODEL ---
model.save('model_lite.keras')
print("--- 9. Lightweight model 'model_lite.keras' saved! ---")

# --- 12. DOWNLOAD THE MODEL ---
print("Downloading your model file... Please wait.")
files.download('model_lite.keras')