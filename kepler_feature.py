import numpy as np
import os

# Verzeichnisse
TRAIN_DIR = "kepler_data/train"
VAL_DIR = "kepler_data/val"
TEST_DIR = "kepler_data/test"

# Funktion zur Überprüfung der Lichtkurvenlängen
def check_lightcurve_lengths(directory, num_samples=30):
    print(f"\n📂 Überprüfung von {directory}:")
    files = sorted(os.listdir(directory))[:num_samples]  # Nur einige zufällige Lichtkurven prüfen
    
    for file in files:
        data = np.load(os.path.join(directory, file), allow_pickle=True).item()
        lightcurve = data["lightcurve"]
        print(f"📊 {file}: {len(lightcurve)} Datenpunkte")

# Überprüfe die Anzahl der Datenpunkte
check_lightcurve_lengths(TRAIN_DIR)
check_lightcurve_lengths(VAL_DIR)
check_lightcurve_lengths(TEST_DIR)
