# import sys
# sys.path.append('.')
import os 
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import argparse
import joblib

from scipy import signal
from scipy.stats import variation, entropy
from scipy import stats
import pywt

import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


from swarm.configs import get_cfg_defaults


def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


def get_args():
  parser = argparse.ArgumentParser(description="Vision Transformers")
  parser.add_argument('--dir', type=str, default='default', help='configuration to load')
  args = parser.parse_args()
  return args

def apply_filter(cfg, arr):
  fs = cfg.FEATS.FS
  cutoff_freq =11
  order = 5
  nyquist_freq = 0.5 * fs
  normal_cutoff = cutoff_freq / nyquist_freq
  b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
  filtered_data = signal.filtfilt(b, a, arr)
  return filtered_data

#*****************************************FEATURE COMPUTATION***************************************
def psd_with_bands(cfg, accel_data, band_width=5):
    # Calculate PSD using Welch's method
    # freqs, psd = signal.welch(accel_data, cfg.FEATS.FS, nperseg=cfg.FEATS.FS//2, 
    #                           noverlap=None, scaling='density')
    freqs, psd = signal.welch(accel_data, cfg.FEATS.FS, nperseg=cfg.FEATS.FS * cfg.FEATS.WINDOW, 
                              noverlap=None, scaling='density')

    # Divide the bandwidth into 5Hz bands
    band_edges = np.arange(0, freqs[-1] + band_width, band_width)
    band_powers = []
    
    for i in range(len(band_edges) - 1):
      low = band_edges[i]
      high = band_edges[i+1]
      
      # Find indices corresponding to the frequency band
      idx_band = np.logical_and(freqs >= low, freqs <= high)
      
      # Extract PSD values within the band
      psd_band = psd[idx_band]
      freqs_band = freqs[idx_band]
      
      # Compute metrics
      avg_power = np.mean(psd_band)
      rms_value = np.sqrt(np.mean(np.square(psd_band)))
      max_value = np.max(psd_band)

      band_powers.append([avg_power, rms_value, max_value])
    
    return band_powers

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def ten_point_average(x):
    return np.convolve(x, np.ones(10), 'valid') / 10

def wavelet_analysis(data, wavelet, scales):
    coeffs = []
    for scale in scales:
        if wavelet == 'morl':
            wavelet_data, _ = pywt.ContinuousWavelet(f'cmor5.0-{scale}').wavefun(level=np.log2(len(data)))
            # wavelet_data2 = signal.morlet(len(data), w=5.0, s=scale)
        else:
            wavelet_data = pywt.Wavelet(wavelet).wavefun(level=scale)[0]
        
        # Pad the wavelet data to match the length of the input data
        if len(wavelet_data) < len(data):
            wavelet_data = np.pad(wavelet_data, (0, len(data) - len(wavelet_data)))
        else:
            wavelet_data = wavelet_data[:len(data)]
        
        coeff = np.convolve(data, wavelet_data, mode='same')
        coeffs.append(coeff)
    
    return coeffs

def compute_wavelet(cfg, data):
  wavelets = ['morl', 'db6', 'db10']
  scales = [4, 5]
  feats = []
  for i, wavelet in enumerate(wavelets):
    coeffs = wavelet_analysis(data, wavelet, scales)
    for j, scale in enumerate(scales):
        rms_value = np.real(rms(coeffs[j]))
        avg_value = np.real(np.mean(ten_point_average(coeffs[j])))
        feats += [rms_value, avg_value]
  return feats

def compute_time_statistics(signal):
  # Convert signal to numpy array if it's not already
  signal = np.array(signal)
  
  # Maximum Value
  max_value = np.max(signal)
  
  # Minimum Value
  min_value = np.min(signal)
  
  # Mean Value
  mean_value = np.mean(signal)
  
  # RMS Value
  rms_value = np.sqrt(np.mean(np.square(signal)))
  
  # Peak-to-Peak Value
  peak_to_peak = max_value - min_value

  # 10-point avg
  ten_point_avg = np.mean(ten_point_average(signal))

  # Kurtosis
  kurtosis = stats.kurtosis(signal)

  return [max_value, min_value, mean_value, rms_value, peak_to_peak, ten_point_avg, kurtosis]



def compute_features(cfg, windows, cols = ["accelUserZFiltered"]):
  # CREATE INPUT DATAFRAME
  columns = []
  for col in cols:
    if "info" in cfg.FEATS.FEATS_LST:
      columns.append(f"cv_{col}")
      columns.append(f"h_{col}")
    if "psd" in cfg.FEATS.FEATS_LST:
      for i in range(4):
        columns.append(f"avg_psd_{i}_{col}")
        columns.append(f"rms_psd_{i}_{col}")
        columns.append(f"max_psd_{i}_{col}")
    if "wavelet" in cfg.FEATS.FEATS_LST:
      wavelets = ['morl', 'db6', 'db10']
      scales = [4, 5]
      for wave in wavelets:
        for scale in scales:
          columns.append(f"rms_{scale}_{wave}_{col}")
          columns.append(f"avg_{scale}_{wave}_{col}")
    if "time" in cfg.FEATS.FEATS_LST:
      columns.append(f"max_{col}")
      columns.append(f"min_{col}")
      columns.append(f"mean_{col}")
      columns.append(f"rms_{col}")
      columns.append(f"ptp_{col}")
      columns.append(f"ten_{col}")
      columns.append(f"kurt_{col}")

  input_columns = columns.copy()
  columns += ["fname"]
  df_all = pd.DataFrame(columns=columns)

  # APPEND DATAFRAME FEATURES PER WINDOW
  for i, sub_df in enumerate(tqdm(windows)):
    sub_stft_feats_lst = []
    ftemp = f"{sub_df['fname'].iloc[0]}_{i}"
    for col in cols:
      z_arr = sub_df[col].to_numpy()
      f, t, Zxx = signal.stft(z_arr, cfg.FEATS.FS, nperseg=cfg.FEATS.FS//2, noverlap=None)

      # WEIGHTED CV and H
      cv_t = variation(np.abs(Zxx), axis=1)
      h_t = entropy(np.abs(Zxx), axis=1)

      band_powers = psd_with_bands(cfg, z_arr)
      wavelet_feats = compute_wavelet(cfg, z_arr)
      time_feats = compute_time_statistics(z_arr)
      
      energy_t = np.sum(np.abs(Zxx)**2, axis=1)
      energy_norm = energy_t.sum()
      energy_w = energy_t/energy_norm

      cv = (energy_w * cv_t).sum()
      h = (energy_w * h_t).sum()

      if "info" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += [cv, h]

      if "psd" in cfg.FEATS.FEATS_LST:
        for band_feat in band_powers:
          sub_stft_feats_lst += band_feat
      
      if "wavelet" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += wavelet_feats

      if "time" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += time_feats
      
    sub_stft_feats_lst.append(ftemp)
    df_all.loc[i] = sub_stft_feats_lst

  return df_all, input_columns

def sliding_window(df, cfg):
    indices_list = []
    # Create a list to store the windows
    windows = []
    # Use the rolling method to create the windows
    window_size = cfg.FEATS.FS * cfg.FEATS.WINDOW
    step_size = int(cfg.FEATS.FS * cfg.FEATS.WINDOW * (1 - cfg.FEATS.OVERLAP))
    print(window_size, step_size, len(df))
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        windows.append(window)
        indices_list.append([i, i+window_size])
    
    return windows, indices_list

def preprocess(cfg, df):
    # Apply filtering
  df['timestamp'] = pd.to_datetime(df['timestamp'], format="ISO8601")
  df = df.sort_values(by='timestamp')
  # df = df[(df['speed'] > 5) & (df['speed'] <= 80)]
  df = df[(df['speed'] > 0)]

  df['accelUserZ'] = df['accelUserZ'].ewm(com=0.4, adjust=False).mean()
  df['gyroZ'] = df['gyroZ'].ewm(com=0.4, adjust=False).mean()

  # High-pass filter
  df['accelUserXFiltered'] = apply_filter(cfg, df['accelUserX'].to_numpy())
  df['accelUserYFiltered'] = apply_filter(cfg, df['accelUserY'].to_numpy())
  df['accelUserZFiltered'] = apply_filter(cfg, df['accelUserZ'].to_numpy())
  df['gyroZFiltered'] = apply_filter(cfg, df['gyroZ'].to_numpy())
  df = df.dropna()

  # DATATYPE CONVERSION
  df['lat'] = df['lat'].astype(float)
  df['lon'] = df['lon'].astype(float)
  return df


def load_data(cfg, root_dir):
  list_df = []

  for entry in os.scandir(root_dir):
    fpath = entry.path 
    fname = entry.name 
    if fname.split(".")[-1] != "csv":
      continue
    df_temp = pd.read_csv(fpath)
    df_temp['fname'] = fname.split(".")[0]
    list_df.append(df_temp)
  df_train = pd.concat(list_df, axis=0, ignore_index=True)
  df_train = preprocess(cfg, df_train)
  return df_train

def save_labels(df, df_inference, output_save_path):
  df = df.reset_index()
  df['prediction'] = "smooth"
  for i, (start, end) in enumerate(df_inference.indices_lst):
    df.loc[start:end, 'prediction'] = df_inference.loc[i, 'predictions']
  df.to_csv(output_save_path)
  return df



class InferTelemetry:
  def __init__(self, model_dir, config_dir, destination_dir) -> None:
    self.model_dir = model_dir
    self.config_dir = config_dir
    self.dest_dir = destination_dir

  def predict(self, csv_dir):

    city = 'tampa'
    configs = ['tampa_31', 'tampa_32']

    y_preds_all = []
    for conf in configs:
      config_path = os.path.join(self.config_dir, city, f"{conf}.yaml")

      # LOAD CONFIGURATION
      cfg = get_cfg_defaults()
      cfg.merge_from_file(config_path)
      cfg.freeze()
      print(cfg)

      inference_paths = [csv_dir]
      dest_csv_name_lst = csv_dir.split("/")
      dest_csv_name = next((item for item in reversed(dest_csv_name_lst) if item), None)

      df_inference = []
      for path in inference_paths:
        df_tr = load_data(cfg, path)
        df_inference.append(df_tr)
      df_inference = pd.concat(df_inference, axis=0, ignore_index=True)
      df_inference_raw = df_inference.sort_values(by='timestamp')


      windows_inference, indices_lst_train = sliding_window(df_inference_raw, cfg)

      cols = cfg.FEATS.COLS
      df_inference, input_columns = compute_features(cfg, windows_inference, cols = cols)
      df_inference['indices_lst'] = indices_lst_train

      clf_path = os.path.join(self.model_dir, f"XGB_{cfg.TRAIN.MODE}_{conf.split('_')[1]}.pkl")

      clf = joblib.load(clf_path)

      transfomers = Pipeline(steps=[
        ('scaler', MinMaxScaler()), 
        ('imputer', SimpleImputer(strategy='mean')),
      ])  

      preprocessor = ColumnTransformer(transformers=[
        ('scale', transfomers, input_columns)
      ])

      model = Pipeline(steps=[
        ('preprocess', preprocessor), 
        ('model', clf)
      ])

      y_preds = clf.predict_proba(df_inference[input_columns])
      y_preds_all.append(y_preds)

    all_probs = np.stack(y_preds_all, axis=0)
    conf, r, p = all_probs.shape


    max_vals = np.max(all_probs, axis=-1)
    max_args = np.argmax(all_probs, axis=-1)

    max_config_vals = np.max(max_vals, axis=0)
    max_config_args = np.argmax(max_vals, axis=0)

    second_dim_indices = np.arange(r)
    final_preds = max_args[max_config_args, second_dim_indices]

    conditions = {
      0: 'Smooth', 
      1: 'Short Distress',
      2: 'Long Distress'
    }
    
    df_inference['predictions'] = final_preds
    df_inference['predictions'] = df_inference['predictions'].apply(lambda x: conditions[x])

    output_save_path = os.path.join(self.dest_dir, f"{dest_csv_name}.csv")
    print(f"Saving at location {output_save_path}")

    save_labels(df_inference_raw, df_inference, output_save_path)

