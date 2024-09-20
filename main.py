import swarm


model_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/checkpoints/" # MODEL LOCATION
dest_dir = "./data/" # DESTINATION DIR
config_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/configs/experiments" # CONFIGURATIONS
csv_dir = "/home/saandeepaath-admin/projects/internship/SWARM/exp1/data/raw/test/tampa_0628_gt"


swarm_obj = swarm.InferTelemetry(model_dir, config_dir)
df_inference, df_inference_raw = swarm_obj.predict(csv_dir)
print(df_inference.shape, df_inference_raw.shape)
