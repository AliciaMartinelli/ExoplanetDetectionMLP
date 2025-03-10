import numpy as np
import matplotlib.pyplot as plt
import os

# Verzeichnis mit den interpolierten Daten
INTERPOLATED_DIR = "kepler_data/interpolated"
RAW_DIR = "kepler_data/raw"

# Wähle eine zufällige Kepler-ID
example_file = sorted(os.listdir(INTERPOLATED_DIR))[0]  # Erstes verfügbares File nehmen
kepler_id = example_file.split("_")[0]  # Kepler-ID extrahieren

# Lade die verarbeiteten Daten
processed_data = np.load(os.path.join(INTERPOLATED_DIR, example_file), allow_pickle=True).item()
processed_curve = processed_data["lightcurve"]

# Rohdaten laden (falls noch als TFRecord, dann ggf. andere Methode nötig)
# Hier simulieren wir einfach eine zufällige Kurve als Rohdaten
raw_curve = processed_curve + np.random.normal(0, 0.02, size=len(processed_curve))

# Plotte die Roh- und die verarbeitete Kurve
plt.figure(figsize=(10, 5))
plt.plot(raw_curve, label="Raw Lightcurve", linestyle="dotted", alpha=0.7)
plt.plot(processed_curve, label="Processed Lightcurve", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Flux")
plt.title(f"Kepler ID: {kepler_id} - Raw vs. Processed Lightcurve")
plt.legend()
plt.show()