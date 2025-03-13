
import numpy as np
import os
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

TRAIN_DIR = "kepler_data/train"
VAL_DIR = "kepler_data/val"
TEST_DIR = "kepler_data/test"

def resample_lightcurve(lightcurve, original_interval=30, target_interval=60):
    time = np.arange(0, len(lightcurve) * original_interval, original_interval)
    interp_func = interp1d(time, lightcurve, kind='linear', fill_value='extrapolate')
    resampled_time = np.arange(0, time[-1], target_interval)
    resampled_curve = interp_func(resampled_time)
    return resampled_curve

def extract_tsfresh_features(directory):

    feature_data = []
    labels = []
    kepler_ids = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        data = np.load(file_path, allow_pickle=True).item()
        
        if "lightcurve" not in data:
            print(f"❌ Fehler: 'lightcurve' nicht gefunden in {file}, wird übersprungen!")
            continue
        
        lightcurve = data["lightcurve"]
        global_view = lightcurve[:2001]
        local_view = lightcurve[2001:]

        global_view_resampled = resample_lightcurve(global_view)
        local_view_resampled = resample_lightcurve(local_view)

        df_global = pd.DataFrame({"id": [file] * len(global_view_resampled), "time": range(len(global_view_resampled)), "flux": global_view_resampled})
        df_local = pd.DataFrame({"id": [file] * len(local_view_resampled), "time": range(len(local_view_resampled)), "flux": local_view_resampled})


        features_global = extract_features(df_global, column_id="id", column_sort="time", n_jobs=0)
        features_local = extract_features(df_local, column_id="id", column_sort="time", n_jobs=0)

        combined_features = pd.concat([features_global, features_local], axis=1)

        feature_data.append(combined_features)
        labels.append(data["label"])
        kepler_ids.append(data["kepler_id"])
    
    feature_df = pd.concat(feature_data)
    feature_df = impute(feature_df)
    
    return feature_df, np.array(labels), np.array(kepler_ids)


train_features, train_labels, train_kepler_ids = extract_tsfresh_features(TRAIN_DIR)
val_features, val_labels, val_kepler_ids = extract_tsfresh_features(VAL_DIR)
X_train = pd.concat([train_features, val_features])
y_train = np.concatenate([train_labels, val_labels])

X_test, y_test, test_kepler_ids = extract_tsfresh_features(TEST_DIR)


scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


np.save("X_train_kepler.npy", X_train_scaled)
np.save("y_train_kepler.npy", y_train)
np.save("X_test_kepler.npy", X_test_scaled)
np.save("y_test_kepler.npy", y_test)

print("DONE!")