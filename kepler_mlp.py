
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

X_train = np.load("X_train_features.npy")
X_test = np.load("X_test_features.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_resampled, return_counts=True)
print("📊 Neue Verteilung nach SMOTE:", dict(zip(unique, counts)))

input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Dense(512, activation="relu", input_shape=(input_dim,)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

model.save("exoplanet_classifier.keras")
model.save("exoplanet_classifier.h5")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss-Verlauf")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy-Verlauf")

    plt.show()

plot_training_history(history)

y_pred = (model.predict(X_test) > 0.55).astype("int32")

print("\n🔹 **Classification Report:**")
print(classification_report(y_test, y_pred))

print("\n🔹 **Confusion Matrix:**")
print(confusion_matrix(y_test, y_pred))