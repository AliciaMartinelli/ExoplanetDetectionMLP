import numpy as np

# Lade die ursprünglichen Datensätze
X_train = np.load("X_train_kepler.npy")
X_test = np.load("X_test_kepler.npy")

# Finde Spalten, die in beiden Sets nur Nullwerte enthalten
null_spalten_train = np.all(X_train == 0, axis=0)
null_spalten_test = np.all(X_test == 0, axis=0)

# Bestimme die gemeinsamen Null-Spalten in beiden Sets
gemeinsame_null_spalten = np.logical_and(null_spalten_train, null_spalten_test)

# Entferne diese Spalten aus den Datensätzen
X_train_cleaned = X_train[:, ~gemeinsame_null_spalten]
X_test_cleaned = X_test[:, ~gemeinsame_null_spalten]

# Speichere die bereinigten Datensätze als neue Dateien
np.save("X_train_cleaned.npy", X_train_cleaned)
np.save("X_test_cleaned.npy", X_test_cleaned)

print(f"✅ Bereinigung abgeschlossen!")
print(f"🔹 Ursprüngliche Anzahl der Features: {X_train.shape[1]}")
print(f"🔹 Anzahl der entfernten Null-Spalten: {np.sum(gemeinsame_null_spalten)}")
print(f"🔹 Neue Anzahl der Features: {X_train_cleaned.shape[1]}")
