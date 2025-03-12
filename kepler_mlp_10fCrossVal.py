import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
import time
import random
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Lade die Trainings- und Testdaten
X_train = np.load("X_train_kepler_cleaned.npy")
X_test = np.load("X_test_kepler_cleaned.npy")
y_train = np.load("y_train_kepler.npy")
y_test = np.load("y_test_kepler.npy")

# Kombiniere Trainings- und Testdaten für Cross-Validation
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Definiere 10-Fold Cross-Validation
k_folds = 10
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Listen zum Speichern der Scores
auc_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

start_time = time.time()  # ⏳ Startzeit messen


# K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🔹 Training Fold {fold + 1}/{k_folds}...")

    # Split in Training- und Validierungsdaten
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Berechne Klassen-Gewichte für Exoplaneten (weil selten)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_fold), y=y_train_fold)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # Modell definieren
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
        layers.Dropout(0.4),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Early Stopping für stabileres Training
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Trainiere das Modell für den aktuellen Fold
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=64,
              validation_data=(X_val_fold, y_val_fold),
              class_weight=class_weight_dict,
              callbacks=[early_stopping], verbose=0)

    # Vorhersagewahrscheinlichkeiten holen
    y_pred_proba = model.predict(X_val_fold).ravel()

    # Berechne ROC-Kurve und optimalen Threshold
    fpr, tpr, thresholds = roc_curve(y_val_fold, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"🔹 Optimaler Threshold für Fold {fold + 1}: {optimal_threshold:.4f}")

    # Klassifiziere mit optimalem Threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

    # Berechne AUC-Score
    auc_score = roc_auc_score(y_val_fold, y_pred_proba)
    auc_scores.append(auc_score)

    # Berechne Precision, Recall, F1-Score
    precision = precision_score(y_val_fold, y_pred_optimal)
    recall = recall_score(y_val_fold, y_pred_optimal)
    f1 = f1_score(y_val_fold, y_pred_optimal)

    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    print(f"✅ AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"🔹 Zeit für eine Kombination: {elapsed_time:.2f} Sekunden")
# Durchschnittliche Werte über alle Folds
mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1 = np.mean(f1_scores)

print(f"\n🏆 Durchschnittliche Scores über {k_folds} Folds:")
print(f"✅ AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"✅ Precision: {mean_precision:.4f}")
print(f"✅ Recall: {mean_recall:.4f}")
print(f"✅ F1-Score: {mean_f1:.4f}")
