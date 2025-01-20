import swarm


def save_labels(df, df_inference):
  df = df.reset_index()
  df['prediction'] = "Smooth"
  for i, (start, end) in enumerate(df_inference.indices_lst):
    df.loc[start:end, 'prediction'] = df_inference.loc[i, 'predictions']
  return df

model_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/checkpoints/" # MODEL LOCATION
dest_dir = "./data/" # DESTINATION DIR
config_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/configs/experiments" # CONFIGURATIONS
csv_dir = [
  '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/20241010_14120019H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_123648H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_123748H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_135740H.csv',
]

onnx_dir = "/home/saandeepaath-admin/projects/internship/SWARM/time_inference/checkpoint/feature_extractor.onnx"
gmm_dir = "/home/saandeepaath-admin/projects/internship/SWARM/time_inference/checkpoint/gmm.joblib"


swarm_obj = swarm.InferTelemetry(model_dir, config_dir)
df_inference, df_inference_raw = swarm_obj.predict(csv_dir)
print(df_inference.shape, df_inference_raw.shape)
print(df_inference_raw.columns)
df_inference_raw = save_labels(df_inference_raw, df_inference)

df_smooth_raw = df_inference_raw[df_inference_raw['prediction'] == "Smooth"]
swarmF_obg = swarm.InferFourth(onnx_dir, gmm_dir)
df_smooth_inference = swarmF_obg.predict(df_smooth_raw)
print(df_inference_raw.shape)

