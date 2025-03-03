""" import lightkurve as lk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.timeseries import LombScargle

# 1. Datenabruf: Lade Lightcurve von Kepler
# Wir wählen eine Beispielquelle (Kepler-10)
target = "Kepler-10"
lc_collection = lk.search_lightcurve(target, mission="Kepler", author="Kepler", cadence="long").download_all()

# Anzahl der verfügbaren Lightcurves anzeigen
num_lcs = len(lc_collection)
print(f"Gefundene Lightcurves für {target}: {num_lcs}")

# Erstelle Ordnerstruktur
base_dir = "kepler10"
images_dir = os.path.join(base_dir, "images")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Lightcurves als .fits speichern und als Bild plotten
for i, lc in enumerate(lc_collection):
    filename_fits = os.path.join(plots_dir, f"{target}_lightcurve_{i+1}.fits")
    filename_png = os.path.join(images_dir, f"{target}_lightcurve_{i+1}.png")
    
    lc.to_fits(filename_fits, overwrite=True)
    print(f"Gespeichert: {filename_fits}")
    
    # Lightcurve plotten und speichern
    plt.figure()
    lc.plot()
    plt.savefig(filename_png)
    plt.close()
    print(f"Gespeichert: {filename_png}")

# Zusammenfassung ausgeben
print(f"Alle {num_lcs} Lightcurves wurden gespeichert und geplottet.") """


import tensorflow as tf
import matplotlib.pyplot as plt

# Pfad zur TFRecord-Datei
tfrecord_file = "kepler/test-00000-of-00001"

# Funktion zum Parsen der TFRecord-Datei
def _parse_function(proto):
    # Definiere die Features basierend auf der erkannten Struktur
    keys_to_features = {
        'global_view': tf.io.FixedLenFeature([2001], tf.float32),
        'local_view': tf.io.FixedLenFeature([201], tf.float32),
        'av_training_set': tf.io.FixedLenFeature([], tf.string),  # Label als String
        'tce_period': tf.io.FixedLenFeature([], tf.float32),
        'tce_duration': tf.io.FixedLenFeature([], tf.float32),
        'tce_depth': tf.io.FixedLenFeature([], tf.float32),
        'tce_model_snr': tf.io.FixedLenFeature([], tf.float32),
        'kepid': tf.io.FixedLenFeature([], tf.int64),
    }
    # Parse das tf.Example-Proto
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    return (parsed_features['global_view'], parsed_features['local_view'], 
            parsed_features['av_training_set'], parsed_features['tce_period'],
            parsed_features['tce_duration'], parsed_features['tce_depth'], 
            parsed_features['tce_model_snr'], parsed_features['kepid'])

# Lade das TFRecordDataset
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(_parse_function)

# Zeige die ersten 5 Lightcurves als Plots
for global_view, local_view, label, period, duration, depth, snr, kepid in dataset.take(5):
    label = label.numpy().decode("utf-8")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Global View plotten
    axes[0].plot(global_view.numpy(), color="blue")
    axes[0].set_title(f"Global View - Kepler ID: {kepid.numpy()} ({label})")
    axes[0].set_xlabel("Zeitpunkte")
    axes[0].set_ylabel("Helligkeit")
    
    # Local View plotten
    axes[1].plot(local_view.numpy(), color="red")
    axes[1].set_title(f"Local View - Kepler ID: {kepid.numpy()} ({label})")
    axes[1].set_xlabel("Zeitpunkte (lokal)")
    
    plt.tight_layout()
    plt.show()
