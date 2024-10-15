import swarm


model_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/checkpoints/" # MODEL LOCATION
dest_dir = "./data/" # DESTINATION DIR
config_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/configs/experiments" # CONFIGURATIONS
csv_dir = [
  '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/20241010_14120019H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_123648H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_123748H.csv',
  # '/home/saandeepaath-admin/projects/internship/SWARM/time_inference/data/tampa_0628_gt/20240628_135740H.csv',
]



swarm_obj = swarm.InferTelemetry(model_dir, config_dir)
df_inference, df_inference_raw = swarm_obj.predict(csv_dir)
print(df_inference.shape, df_inference_raw.shape)
