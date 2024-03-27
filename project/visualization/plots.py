import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_heatmaps(config: dict, results: list, seq=5):
    """
    Plots heatmaps of predictions and ground truth for both upper and surface variables for a given sequence index.
    
    Parameters:
    - config (dict): Dictionary containing the configuration settings.
    - results (list): List of dictionaries containing the predictions and ground truth.
    - seq (int): Sequence index to plot.
    """

    plot_path = os.path.join(config["global"]["path"], config["global"]["reports_path"])

    upper_var_names = config["preprocessing"]["upper_var_names"][1:]
    pressure_levels_upper = config["preprocessing"]["limits"]["levels"][1]
    surface_var_names = config["preprocessing"]["surface_var_names"][1:]
    
    nrows = pressure_levels_upper + 1
    ncols = max(len(upper_var_names), len(surface_var_names)) * 2 + 1
    
    fig = plt.figure(figsize=(20 + 2, 2 * nrows + 1))
    gs = GridSpec(nrows, ncols, figure=fig)
    
    # Plot surface variables
    surface_result = next((r for r in results if r['type'] == 'surface'), None)
    if surface_result:
        for var_index, var_name in enumerate(surface_var_names):
            ax_pred = fig.add_subplot(gs[0, var_index * 2 + 1])
            ax_gt = fig.add_subplot(gs[0, var_index * 2 + 2])
            
            prediction = surface_result['prediction'][seq, :, :, var_index]
            ground_truth = surface_result['ground_truth'][seq, :, :, var_index]
            
            ax_pred.imshow(prediction, cmap='hot', interpolation='nearest')
            ax_pred.axis('off')
            ax_gt.imshow(ground_truth, cmap='hot', interpolation='nearest')
            ax_gt.axis('off')
            
            # Set titles for surface variables
            ax_pred.set_title(f'Pred {var_name}')
            ax_gt.set_title(f'GT {var_name}')
                
        ax_surface = fig.add_subplot(gs[0, 0])
        ax_surface.text(0.5, 0.5, "Surface", fontsize=12, ha='center', va='center')
        ax_surface.axis('off')
    else:
        print("Surface results not found.")

    # Plot upper variables
    upper_result = next((r for r in results if r['type'] == 'upper'), None)
    if upper_result:
        for level in range(pressure_levels_upper):
            for var_index, var_name in enumerate(upper_var_names):
                ax_pred = fig.add_subplot(gs[level + 1, var_index * 2 + 1])
                ax_gt = fig.add_subplot(gs[level + 1, var_index * 2 + 2])
                
                prediction = upper_result['prediction'][seq, level, :, :, var_index]
                ground_truth = upper_result['ground_truth'][seq, level, :, :, var_index]
                
                ax_pred.imshow(prediction, cmap='hot', interpolation='nearest')
                ax_pred.axis('off')
                ax_gt.imshow(ground_truth, cmap='hot', interpolation='nearest')
                ax_gt.axis('off')
                
                # Set titles for upper variables
                if level == 0: 
                    ax_pred.set_title(f'Pred {var_name}')
                    ax_gt.set_title(f'GT {var_name}')
                    
            ax_level = fig.add_subplot(gs[level + 1, 0])
            ax_level.text(0.5, 0.5, f'Level {level + 1}', fontsize=12, ha='center', va='center')
            ax_level.axis('off')
    else:
        print("Upper results not found.")
    
    plt.suptitle(f'\nPredictions and Ground Truth at sequence time {seq+1}\n', fontsize=16)
    plt.tight_layout()
    plt.show()

    if config["plots"]["pred_vs_true"]:
        plt.savefig(os.path.join(plot_path, f'{config["train"]["model_name"]}_heatmaps_{seq+1}.png'))

    return fig
