
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import os 
import pandas as pd

import librosa
from sklearn.mixture import GaussianMixture
from sklearn.base import clone
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, \
  precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix, make_scorer, classification_report

import xgboost as xgb

from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks


from scipy import signal
from scipy.stats import variation, entropy
from scipy import stats
import pywt
import shap
from imports import *
import joblib

from tqdm import tqdm
import numpy as np



def sliding_window(df, cfg):
    indices_list = []
    # Create a list to store the windows
    windows = []
    # Use the rolling method to create the windows
    window_size = cfg.FEATS.FS * cfg.FEATS.WINDOW
    step_size = int(cfg.FEATS.FS * cfg.FEATS.WINDOW * (1 - cfg.FEATS.OVERLAP))
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        windows.append(window)
        indices_list.append([i, i+window_size])
    
    return windows, indices_list

def apply_filter(cfg, arr):
  fs = cfg.FEATS.FS
  cutoff_freq =5
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

      # feats = compute_time_statistics(psd_band)

      band_powers.append([avg_power, rms_value, max_value])
      # band_powers.append([avg_power, rms_value, max_value] + feats)
    
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


def compute_stft_statistics(stft_array):
  frequencies = np.linspace(0, 0.5, stft_array.shape[0])
  magnitudes = np.abs(stft_array)

  # 1. Spectral centroid
  spectral_centroid = np.sum(frequencies[:, np.newaxis] * magnitudes, axis=0) / (np.sum(magnitudes, axis=0)  + 1e-10)

  # 2. Spectral bandwidth
  spectral_bandwidth = np.sqrt(np.sum((frequencies[:, np.newaxis] - spectral_centroid) ** 2 * magnitudes, axis=0) / (np.sum(magnitudes, axis=0)  + 1e-10)) 

  # 3. Spectral flux
  spectral_flux = np.sqrt(np.sum(np.diff(magnitudes, axis=0)**2, axis=0))

  # 4. Spectral rolloff
  rolloff_percentage = 0.85
  rolloff_thresholds = rolloff_percentage * np.sum(magnitudes, axis=0)
  rolloff_indices = np.argmax(np.cumsum(magnitudes, axis=0) >= rolloff_thresholds, axis=0)
  spectral_rolloff = frequencies[rolloff_indices]

  # # 8. Spectral entropy
  probability_spectrum = magnitudes / (np.sum(magnitudes, axis=0)  + 1e-10)
  probability_spectrum[probability_spectrum == 0] = 1e-10  # To avoid log(0)
  spectral_entropy = -np.sum(probability_spectrum * np.log(probability_spectrum), axis=0)

  # 9. Dominant frequency
  dominant_frequency_indices = np.argmax(magnitudes, axis=0)
  dominant_frequencies = frequencies[dominant_frequency_indices]

  # 11. Harmonic ratio
  harmonic_ratio = np.sum(magnitudes[1:10, :], axis=0) / (np.sum(magnitudes, axis=0) + 1e-10)

  # 12. Pitch
  pitches, _ = librosa.piptrack(S=stft_array)
  pitch = np.max(pitches, axis=0)
  pitch = pitch[~np.isnan(pitch)]

  return compute_time_statistics(harmonic_ratio) + \
          compute_time_statistics(spectral_flux) + \
            compute_time_statistics(spectral_bandwidth) + \
              compute_time_statistics(dominant_frequencies) 
    



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




#*****************************************I/O Operation***************************************


def preprocess(cfg, df):
    # Apply filtering
  df['timestamp'] = pd.to_datetime(df['timestamp'], format="ISO8601")
  df = df.sort_values(by='timestamp')
  # df = df[(df['speed'] > 5) & (df['speed'] <= 80)]
  df = df[(df['speed'] > 0)]

  # df['accelUserX'] = df['accelUserX'].ewm(span=17, adjust=False).mean()
  # df['accelUserZ'] = df['accelUserZ'].ewm(span=17, adjust=False).mean()
  # df['gyroZ'] = df['gyroZ'].ewm(span=17, adjust=False).mean()

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

def load_data(cfg, root_dir, test_set_names=[], mode="train"):
  list_df = []

  for entry in os.scandir(root_dir):
    fpath = entry.path 
    fname = entry.name 
    if fname.split(".")[-1] != "csv":
      continue
    
    fname_wo = fname.split(".")[0]
    df_temp = pd.read_csv(fpath)
    df_temp['fname'] = fname.split(".")[0]
    if fname_wo in test_set_names or mode == "test":
      continue

    # # NOTE: TESTING ONLY
    # if "tampa_0725_processed" in fname:
    #   print("skipping...")
    #   continue
    # else:
    #   list_df.append(df_temp)
    list_df.append(df_temp)
    
  df_train = pd.concat(list_df, axis=0, ignore_index=True)
  df_train = preprocess(cfg, df_train)
  return df_train

def compute_features(cfg, windows, mode="train", cols = ["accelUserZFiltered"]):
  # CREATE INPUT DATAFRAME
  columns = []
  for col in cols:
    if "info" in cfg.FEATS.FEATS_LST:
      # columns.append(f"cv_{col}")
      # columns.append(f"h_{col}")

      for i in range(7):
        columns.append(f"entropy_{i}_{col}")
        columns.append(f"flux_{i}_{col}")
        columns.append(f"bandwidth_{i}_{col}")
        columns.append(f"freq_{i}_{col}")
        
    if "psd" in cfg.FEATS.FEATS_LST:
      for i in range(4):
        columns.append(f"avg_psd_{i}_{col}")
        columns.append(f"rms_psd_{i}_{col}")
        columns.append(f"max_psd_{i}_{col}")
        # for j in range(7):
        #   columns.append(f"time_psd_{i}_{j}_{col}")

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
  if mode == "test":
    columns += ["label"]
  df_all = pd.DataFrame(columns=columns)

  window_length = cfg.FEATS.FS
  overlap_length = cfg.FEATS.FS//2
  hanning_window = np.hanning(window_length)

  # APPEND DATAFRAME FEATURES PER WINDOW
  for i, sub_df in enumerate(tqdm(windows)):
    sub_stft_feats_lst = []
    ftemp = f"{sub_df['fname'].iloc[0]}_{i}"
    for col in cols:
      z_arr = sub_df[col].to_numpy()
      SFT = signal.ShortTimeFFT(hanning_window, hop=overlap_length, fs=cfg.FEATS.FS)
      Zxx = SFT.stft(z_arr)
      Zxx_abs = np.abs(Zxx)
      # f, t, Zxx = signal.stft(z_arr, cfg.FEATS.FS, nperseg=cfg.FEATS.FS//2, noverlap=None)

      # WEIGHTED CV and H
      cv_t = variation(np.abs(Zxx), axis=1)
      h_t = entropy(np.abs(Zxx), axis=1)

      band_powers = psd_with_bands(cfg, z_arr)
      wavelet_feats = compute_wavelet(cfg, z_arr)
      time_feats = compute_time_statistics(z_arr)
      freq_feats = compute_stft_statistics(Zxx)

      # for i in freq_feats:
      #   print(min(i), max(i))
      # e()
      
      energy_t = np.sum(np.abs(Zxx)**2, axis=1)
      energy_norm = energy_t.sum()
      energy_w = energy_t/energy_norm

      cv = (energy_w * cv_t).sum()
      h = (energy_w * h_t).sum()

      if "info" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += freq_feats

      if "psd" in cfg.FEATS.FEATS_LST:
        for band_feat in band_powers:
          sub_stft_feats_lst += band_feat
      
      if "wavelet" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += wavelet_feats

      if "time" in cfg.FEATS.FEATS_LST:
        sub_stft_feats_lst += time_feats
      
    avg_speed = sub_df['speed'].mean()

    sub_stft_feats_lst.append(ftemp)
    if mode == "test":
      label = sub_df['label'].loc[sub_df['label'] != 'smooth'].iloc[0] if any(sub_df['label'] != 'smooth') else 'smooth'
      sub_stft_feats_lst += [label]

      # if label == "short_distress":
      #   fig = px.line(sub_df, x="timestamp", y=['accelUserZFiltered'], title=f"{label}")
      #   fig.write_image(f"{line_viz_dir}/{ftemp}.png", width=1080, height=1080, scale=1)

      
    df_all.loc[i] = sub_stft_feats_lst

  return df_all, input_columns