import numpy as np
import os
from scipy.signal import correlate
from scipy.stats import entropy
import pywt
from astropy.timeseries import LombScargle  # Lomb-Scargle für Frequenzanalyse

# Verzeichnisse mit den gespeicherten Lichtkurven
data_dir = "kepler_data"
train_files = [os.path.join(data_dir, "train", f) for f in os.listdir(os.path.join(data_dir, "train")) if f.endswith(".npy")]
val_files = [os.path.join(data_dir, "val", f) for f in os.listdir(os.path.join(data_dir, "val")) if f.endswith(".npy")]
test_files = [os.path.join(data_dir, "test", f) for f in os.listdir(os.path.join(data_dir, "test")) if f.endswith(".npy")]

# Train und Val zusammenlegen
train_files += val_files

# Initialisieren von Listen für Features und Labels
X_train, y_train = [], []
X_test, y_test = [], []

# Feature-Extraktion Methoden
def extract_features(global_view, local_view):
    features = []
    
    ### 1. GLOBAL VIEW FEATURES ###
    global_features = []
    
    # Differenzielle Filterung (Ableitung)
    diff_global = np.diff(global_view)
    global_features.extend([np.mean(diff_global), np.std(diff_global)])
    
    # Lomb-Scargle Periodogramm
    time = np.arange(len(global_view))  # Simulierte Zeitachse für Lomb-Scargle
    frequency, power = LombScargle(time, global_view).autopower()
    max_power_idx = np.argmax(power)
    global_features.extend([frequency[max_power_idx], power[max_power_idx]])

    # Wavelet-Analyse (Haar-Wavelet)
    wavelet_global = pywt.wavedec(global_view, 'haar', level=3)
    global_features.append(np.mean(wavelet_global[0]))

    # Entropie-Analyse
    global_features.append(entropy(np.histogram(global_view, bins=50, density=True)[0]))

    ### 2. LOCAL VIEW FEATURES ###
    local_features = []

    # Differenzielle Filterung (Ableitung)
    diff_local = np.diff(local_view)
    local_features.extend([np.mean(diff_local), np.std(diff_local)])

    # Wavelet-Analyse (Haar-Wavelet)
    wavelet_local = pywt.wavedec(local_view, 'haar', level=3)
    local_features.append(np.mean(wavelet_local[0]))

    # Entropie-Analyse
    local_features.append(entropy(np.histogram(local_view, bins=50, density=True)[0]))

    # Matched Filtering (synthetische Transit-Form)
    synthetic_transit = np.exp(-np.linspace(-2, 2, len(local_view))**2)  # Gauss-Transit
    correlation = correlate(local_view, synthetic_transit, mode='valid')
    local_features.append(np.max(correlation))

    # Feature-Kombination: Global zuerst, dann Local
    features.extend(global_features + local_features)

    return np.array(features)

# Datenverarbeitung für Training & Test
def process_files(file_list, X_list, y_list):
    for file in file_list:
        data = np.load(file, allow_pickle=True).item()  # .npy-Datei laden
        lightcurve = data["lightcurve"]
        global_view = lightcurve[:2001]
        local_view = lightcurve[2001:]
        label = data["label"]

        y_label = 1 if label == "PC" else 0
        features = extract_features(global_view, local_view)
        X_list.append(features)
        y_list.append(y_label)

# Verarbeitung von Training & Test Daten
process_files(train_files, X_train, y_train)
process_files(test_files, X_test, y_test)

# Umwandeln in NumPy-Arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Speichern der Feature-Daten
np.save("X_train_kepler_feat.npy", X_train)
np.save("X_test_kepler_feat.npy", X_test)
np.save("y_train_kepler_feat.npy", y_train)
np.save("y_test_kepler_feat.npy", y_test)

print("Feature-Extraktion abgeschlossen. Dateien gespeichert.")
