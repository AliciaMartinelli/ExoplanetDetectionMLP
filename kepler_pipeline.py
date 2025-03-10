import tensorflow as tf
import numpy as np
import os
import json
import scipy.interpolate as interp
from scipy.signal import savgol_filter

DATA_DIR = os.path.join(os.path.dirname(__file__), "kepler_data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERPOLATED_DIR = os.path.join(DATA_DIR, "interpolated")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

for folder in [INTERPOLATED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, LOGS_DIR]:
    os.makedirs(folder, exist_ok=True)

log_file = os.path.join(LOGS_DIR, "log.txt")
processed_files = []
skipped_files = []

def log(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

def _parse_function(proto):
    keys_to_features = {
        'global_view': tf.io.FixedLenFeature([2001], tf.float32),
        'local_view': tf.io.FixedLenFeature([201], tf.float32),
        'av_training_set': tf.io.FixedLenFeature([], tf.string),
        'kepid': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['kepid'], parsed_features['global_view'], parsed_features['av_training_set']

def preprocess_lightcurve(time_series):
    time_series = np.array(time_series)
    
    diffs = np.abs(time_series[1:-1] - (time_series[:-2] + time_series[2:]) / 2)
    mask = diffs < (5 * np.std(diffs))
    time_series[1:-1][~mask] = np.median(time_series)
    
    flattened = savgol_filter(time_series, window_length=51, polyorder=3)
    detrended = time_series - flattened
    
    assert len(time_series) == len(detrended), "Fehler: Flattening hat die Länge verändert!"
    
    return detrended

def interpolate_lightcurve(lightcurve):
    x = np.arange(len(lightcurve))
    mask = ~np.isnan(lightcurve)
    interpolator = interp.interp1d(x[mask], lightcurve[mask], kind='linear', fill_value='extrapolate')
    interpolated = interpolator(x)
    
    assert not np.isnan(interpolated).any(), "Fehler: NaN-Werte nach Interpolation gefunden!"
    
    return interpolated

tfrecord_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR)]

total_pc, total_npc = 0, 0
kepler_id_counts = {}

for file in tfrecord_files:
    print(f"📂 Teste Datei: {file}")
    dataset = tf.data.TFRecordDataset(file)

    try:
        for raw_record in dataset.take(1):  # Nur den ersten Datensatz testen
            print(f"✅ Datei {file} ist lesbar!")
    except tf.errors.DataLossError:
        print(f"❌ Fehlerhafte Datei: {file}")
        skipped_files.append(file)
        continue  # Überspringe diese Datei

    dataset = dataset.map(_parse_function)
                
    for kepid, global_view, label in dataset:
        kepid = int(kepid.numpy())
        label = label.numpy().decode("utf-8")
        global_view = global_view.numpy()
        
        y_label = 1 if label == "PC" else 0
        
        # Erzeuge einen einzigartigen Dateinamen, falls die Kepler-ID mehrfach vorkommt
        if kepid in kepler_id_counts:
            kepler_id_counts[kepid] += 1
        else:
            kepler_id_counts[kepid] = 1
        
        unique_filename = f"{kepid}_{kepler_id_counts[kepid]}.npy"
        unique_save_path = os.path.join(INTERPOLATED_DIR, unique_filename)

        cleaned_curve = preprocess_lightcurve(global_view)
        interpolated_curve = interpolate_lightcurve(cleaned_curve)
        
        np.save(unique_save_path, {"lightcurve": interpolated_curve, "label": y_label, "kepler_id": kepid})
        print(f"💾 Gespeichert: {unique_save_path}")
        
        # Bestimme den Zielordner (train, val, test)
        if "train" in file:
            target_dir = TRAIN_DIR
        elif "val" in file:
            target_dir = VAL_DIR
        elif "test" in file:
            target_dir = TEST_DIR
        else:
            continue  # Sollte nie passieren
        
        final_save_path = os.path.join(target_dir, unique_filename)
        os.rename(unique_save_path, final_save_path)
        print(f"📂 Datei verschoben nach: {final_save_path}")
        
        if y_label == 1:
            total_pc += 1
        else:
            total_npc += 1
    
    processed_files.append(file)
    log(f"{file}: Verarbeitung abgeschlossen. PC: {total_pc}, NPC: {total_npc}")

log(f"Gesamt: PC = {total_pc}, NPC = {total_npc}")