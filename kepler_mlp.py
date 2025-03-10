import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# Lade die Trainings- und Testdaten
X_train = np.load("X_train_kepler.npy")
X_test = np.load("X_test_kepler.npy")
y_train = np.load("y_train_kepler.npy")
y_test = np.load("y_test_kepler.npy")

# Berechne Klassen-Gewichte (weil Exoplaneten seltener sind)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"🔹 Klassen-Gewichte: {class_weight_dict}")

# Definiere das MLP-Modell
input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(input_dim,)),
    layers.Dropout(0.25),
    
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.25),

    layers.Dense(32, activation="relu"),

    layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# Early Stopping für stabileres Training
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# Trainiere das Modell
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weight_dict,  # Wichte Exoplaneten höher
                    callbacks=[early_stopping])

# Speichere das trainierte Modell
model.save("exoplanet_classifier.h5")

# Trainingsverlauf visualisieren
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss-Verlauf")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy", color="blue")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy-Verlauf")

    plt.show()

plot_training_history(history)

# Vorhersagewahrscheinlichkeiten holen
y_pred_proba = model.predict(X_test).ravel()

# Berechne ROC-Kurve und optimalen Threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"🔹 Optimaler Threshold aus ROC-Kurve: {optimal_threshold:.4f}")

# Finale Vorhersagen basierend auf optimalem Threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Classification Report
print(f"\n🔹 Classification Report (Optimal Threshold = {optimal_threshold:.4f})")
print(classification_report(y_test, y_pred_optimal))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimal)
print("\n🔹 Confusion Matrix:")
print(cm)

# Confusion Matrix visualisieren
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Optimal Threshold = {optimal_threshold:.4f})")
plt.colorbar()
plt.xticks([0, 1], ["NPC (0)", "PC (1)"])
plt.yticks([0, 1], ["NPC (0)", "PC (1)"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

plt.show()

# AUC ausgeben
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"✅ Genauer AUC-Score: {auc_score:.4f}")
