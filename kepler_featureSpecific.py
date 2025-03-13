import numpy as np
import os
from scipy.signal import correlate
from scipy.stats import entropy
import pywt
from astropy.timeseries import LombScargle

data_dir = "kepler_data"
train_files = [os.path.join(data_dir, "train", f) for f in os.listdir(os.path.join(data_dir, "train")) if f.endswith(".npy")]
val_files = [os.path.join(data_dir, "val", f) for f in os.listdir(os.path.join(data_dir, "val")) if f.endswith(".npy")]
test_files = [os.path.join(data_dir, "test", f) for f in os.listdir(os.path.join(data_dir, "test")) if f.endswith(".npy")]

train_files += val_files

X_train, y_train = [], []
X_test, y_test = [], []
feature_names = []

def extract_features(global_view, local_view):
    features = []
    global_features = []
    local_features = []
    
    ### 1. GLOBAL VIEW FEATURES ###
    
    # Differenzielle Filterung (Ableitung)
    diff_global = np.diff(global_view)
    global_features.append(np.mean(diff_global))
    global_features.append(np.std(diff_global))
    feature_names.extend(["global_diff_mean", "global_diff_std"])
    
    # Lomb-Scargle Periodogramm
    time = np.arange(len(global_view))
    frequency, power = LombScargle(time, global_view).autopower()
    max_power_idx = np.argmax(power)
    global_features.append(frequency[max_power_idx])
    global_features.append(power[max_power_idx])
    feature_names.extend(["global_lomb_freq", "global_lomb_power"])

    # Wavelet-Analyse (Haar-Wavelet)
    wavelet_global = pywt.wavedec(global_view, 'haar', level=3)
    global_features.append(np.mean(wavelet_global[0]))
    feature_names.append("global_wavelet_mean")

    # Entropie-Analyse
    global_features.append(entropy(np.histogram(global_view, bins=50, density=True)[0]))
    feature_names.append("global_entropy")

    ### 2. LOCAL VIEW FEATURES ###
    
    # Differenzielle Filterung (Ableitung)
    diff_local = np.diff(local_view)
    local_features.append(np.mean(diff_local))
    local_features.append(np.std(diff_local))
    feature_names.extend(["local_diff_mean", "local_diff_std"])

    # Wavelet-Analyse (Haar-Wavelet)
    wavelet_local = pywt.wavedec(local_view, 'haar', level=3)
    local_features.append(np.mean(wavelet_local[0]))
    feature_names.append("local_wavelet_mean")

    # Entropie-Analyse
    local_features.append(entropy(np.histogram(local_view, bins=50, density=True)[0]))
    feature_names.append("local_entropy")

    # Matched Filtering (synthetische Transit-Form)
    synthetic_transit = np.exp(-np.linspace(-2, 2, len(local_view))**2)  # Gauss-Transit
    correlation = correlate(local_view, synthetic_transit, mode='valid')
    local_features.append(np.max(correlation))
    feature_names.append("local_matched_filter")

    features.extend(global_features + local_features)

    return np.array(features)


def process_files(file_list, X_list, y_list):
    for file in file_list:
        data = np.load(file, allow_pickle=True).item()
        lightcurve = data["lightcurve"]
        global_view = lightcurve[:2001]
        local_view = lightcurve[2001:]
        label = data["label"]

        features = extract_features(global_view, local_view)
        X_list.append(features)
        y_list.append(label)

process_files(train_files, X_train, y_train)
process_files(test_files, X_test, y_test)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

np.save("X_train_kepler_feat.npy", X_train)
np.save("X_test_kepler_feat.npy", X_test)
np.save("y_train_kepler_feat.npy", y_train)
np.save("y_test_kepler_feat.npy", y_test)

with open("feature_names.txt", "w") as f:
    for name in feature_names:
        f.write(name + "\n")

print("Feature-Extraktion abgeschlossen")

print("\n📊 **Feature-Namen:**")
print("🌍 **Globale Features:**", feature_names[:len(global_features)])
print("🔍 **Lokale Features:**", feature_names[len(global_features):])
