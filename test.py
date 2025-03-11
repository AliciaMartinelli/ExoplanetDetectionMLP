import numpy as np

# Lade die Datensätze
X_train = np.load("X_train_kepler.npy")
X_test = np.load("X_test_kepler.npy")
y_train = np.load("y_train_kepler.npy")
y_test = np.load("y_test_kepler.npy")

# Finde Spalten in X_train, die nur Nullwerte enthalten
zero_columns = np.all(X_train == 0, axis=0)

# Entferne diese Spalten aus X_train und X_test
X_train_cleaned = X_train[:, ~zero_columns]
X_test_cleaned = X_test[:, ~zero_columns]

# Speichere die bereinigten Daten als neue Dateien
np.save("X_train_kepler.npy", X_train_cleaned)
np.save("X_test_kepler.npy", X_test_cleaned)
np.save("y_train_kepler", y_train)  # y hat keine Spalten, bleibt gleich
np.save("y_test_kepler.npy", y_test)    # y hat keine Spalten, bleibt gleich

print(f"✅ Bereinigung abgeschlossen!")
print(f"🔹 Ursprüngliche X_train-Form: {X_train.shape}")
print(f"🔹 Bereinigte X_train-Form: {X_train_cleaned.shape}")
print(f"🔹 Ursprüngliche X_test-Form: {X_test.shape}")
print(f"🔹 Bereinigte X_test-Form: {X_test_cleaned.shape}")
