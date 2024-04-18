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
    project="WindFormer",
    
    notes="Model 7 - First pretrained model",
    tags=["45 years data", "BERT Base", "32 time steps"]
)

wandb_config = wandb.config

assign_config(wandb_config, config)

process_climatology(config)
preprocess_data(config)
normalize_data(config)

train_model(config)

results = infer(config)

rmse = calculate_rmse(results)
wandb.log({"rmse": rmse})

fig = plot_heatmaps(config, results, 31)
wandb.log({"heatmap": fig})

run.finish()