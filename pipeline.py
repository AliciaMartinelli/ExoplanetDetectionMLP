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

# Pfad zur TFRecord-Datei
tfrecord_file = '/kepler/test-00000-of-00001'

# Funktion zum Parsen der TFRecord-Datei
def _parse_function(proto):
    # Definiere die Features, die gelesen werden sollen
    keys_to_features = {
        'global_view': tf.io.FixedLenFeature([2001], tf.float32),
        'local_view': tf.io.FixedLenFeature([201], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    # Parse die Eingabe tf.Example proto mit den definierten Features
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['global_view'], parsed_features['local_view'], parsed_features['label']

# Erstelle einen Dataset-Objekt
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(_parse_function)

# Iteriere über das Dataset
for global_view, local_view, label in dataset:
    print(f'Label: {label.numpy()}')
    # Hier kannst du weitere Verarbeitungsschritte hinzufügen

