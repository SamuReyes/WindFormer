from utils.config_loader import load_config, assign_config
from processing.climatology import process_climatology
from processing.preprocessing import preprocess_data
from processing.normalize import normalize_data
from model.train import train_model
from model.inference import infer, infer_future
from model.metrics import weighted_rmse, weighted_rmse_acc_future
from visualization.plots import plot_heatmaps
import json

config = load_config("./config/params.yaml")

print("Starting inference")

results_future = infer_future(config, 12)

print("Inference finished")

results = {}

for i in range(12):
    print(f"Step {i} of 12")
    rmse_surface, rmse_upper, acc_surface, acc_upper = weighted_rmse_acc_future(config, results_future, i)
    results[i] = {"upper_rmse": rmse_upper, "surface_rmse": rmse_surface, "upper_acc": acc_upper, "surface_acc": acc_surface}

# Guardar los resultados en un archivo JSON
with open('metrics-base-100-epochs.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
