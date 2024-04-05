from utils.config_loader import load_config, assign_config
from processing.climatology import process_climatology
from processing.preprocessing import preprocess_data
from processing.normalize import normalize_data
from model.train import train_model
from model.inference import infer
from model.metrics import calculate_rmse
from visualization.plots import plot_heatmaps

config = load_config("./config/params.yaml")

import wandb
import os

os.environ['WANDB_NOTEBOOK_NAME'] = "pipeline.ipynb"
wandb.login()

run = wandb.init(
    project="WindViVit",
    
    notes="Model 3 - BERT Large params",
    tags=["28 years data", "BERT Large", "Convolutional patching"]
)

wandb_config = wandb.config

assign_config(wandb_config, config)

train_model(config)

results = infer(config)

rmse = calculate_rmse(results)
wandb.log({"upper_rmse": rmse["upper_rmse"], "surface_rmse": rmse["surface_rmse"]})

fig = plot_heatmaps(config, results, 5)
wandb.log({"heatmap": fig})

run.finish()