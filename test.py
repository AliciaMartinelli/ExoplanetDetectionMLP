import numpy as np
import matplotlib.pyplot as plt

# Datei laden
file_path = "kepler_data/test/1429589_1.npy"  # Falls die Datei woanders liegt, den Pfad anpassen
data = np.load(file_path, allow_pickle=True).item()

# Prüfen, ob die Keys existieren
if "lightcurve" in data:
    lightcurve = data["lightcurve"]

    # Global View und Local View aufteilen
    global_view = lightcurve[:2001]
    local_view = lightcurve[2001:]

    # Plot erstellen
    plt.figure(figsize=(12, 5))

    # Global View plotten
    plt.subplot(1, 2, 1)
    plt.plot(global_view, label="Global View", color="blue")
    plt.xlabel("Zeitpunkte")
    plt.ylabel("Flux")
    plt.title("Global View")
    plt.legend()

    # Local View plotten
    plt.subplot(1, 2, 2)
    plt.plot(local_view, label="Local View", color="red")
    plt.xlabel("Zeitpunkte")
    plt.ylabel("Flux")
    plt.title("Local View")
    plt.legend()

    # Anzeigen
    plt.tight_layout()
    plt.show()

else:
    print("❌ Fehler: Key 'lightcurve' nicht in der Datei gefunden.")
