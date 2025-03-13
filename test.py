import numpy as np

files = [
    "X_train_kepler_feat.npy",
    "X_test_kepler_feat.npy",
    "y_train_kepler_feat.npy",
    "y_test_kepler_feat.npy"
]


for file in files:
    try:
        data = np.load(file)
        print(f"📊 Datei: {file}")
        print(f"   ➤ Form: {data.shape}")
        print(f"   ➤ Datentyp: {data.dtype}\n")
    except Exception as e:
        print(f"❌ Fehler beim Laden von {file}: {e}")
