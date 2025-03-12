import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import itertools
import time
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

# Lade die Daten
X = np.concatenate((np.load("X_train_kepler.npy"), np.load("X_test_kepler.npy")))
y = np.concatenate((np.load("y_train_kepler.npy"), np.load("y_test_kepler.npy")))

# Definiere die Parameter für den Grid Search
architectures = [[64, 32, 16], [256, 128, 64]]
learning_rates = [0.0001, 0.001, 0.005]
dropouts = [0.4, 0.5]
batch_sizes = [16, 32, 64]

# Generiere alle Kombinationen
param_combinations = list(itertools.product(architectures, learning_rates, dropouts, batch_sizes))

# Speichert die Ergebnisse
results = []

def build_model(architecture, dropout, learning_rate):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))
    
    for units in architecture:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(dropout))
    
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Starte den Grid Search
for i, (arch, lr, dropout, batch) in enumerate(param_combinations, start=1):
    print(f"\n🔹 Testing Combination {i}/{len(param_combinations)} → LR={lr}, Architecture={arch}, Dropout={dropout}, Batch Size={batch}")
    start_time = time.time()
    
    try:
        # 10-Fold Cross Validation
        fold_aucs, fold_precisions, fold_recalls, fold_f1s = [], [], [], []
        for fold in range(10):
            print(f"  ➤ Training Fold {fold + 1}/10...")
            
            # Split Train/Test für diesen Fold
            val_idx = np.arange(fold, len(X), 10)
            train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Berechne Klassen-Gewichte
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
            
            # Baue das Modell
            model = build_model(arch, dropout, lr)
            
            # Trainiere das Modell
            model.fit(X_train, y_train, epochs=50, batch_size=min(batch, len(X_train)),
                      validation_data=(X_val, y_val), class_weight=class_weight_dict, verbose=0,
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)])
            
            # Vorhersagewahrscheinlichkeiten holen
            y_pred_proba = model.predict(X_val).ravel()
            
            # Berechne ROC-Kurve und optimalen Threshold
            fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Finale Vorhersagen
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Berechne Metriken
            auc_score = roc_auc_score(y_val, y_pred_proba)
            report = classification_report(y_val, y_pred_optimal, output_dict=True)
            
            fold_aucs.append(auc_score)
            fold_precisions.append(report['1']['precision'])
            fold_recalls.append(report['1']['recall'])
            fold_f1s.append(report['1']['f1-score'])
        
        # Speichere Ergebnisse für diese Kombination
        results.append({
            "Architecture": arch, "LR": lr, "Dropout": dropout, "Batch Size": batch,
            "AUC": np.mean(fold_aucs), "Precision": np.mean(fold_precisions),
            "Recall": np.mean(fold_recalls), "F1-Score": np.mean(fold_f1s),
            "Time": time.time() - start_time
        })
        
        print(f"✅ Finished Combination {i}/{len(param_combinations)} → AUC: {np.mean(fold_aucs):.4f}, F1-Score: {np.mean(fold_f1s):.4f}, Time: {time.time() - start_time:.2f}s")
    
    except Exception as e:
        print(f"❌ Fehler bei Kombination {i}: {e}")

# Speichere die Ergebnisse als CSV
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv("grid_search_results.csv", index=False)

print("\n🏆 Grid Search abgeschlossen! Ergebnisse gespeichert in 'grid_search_results.csv'")