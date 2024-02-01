import matplotlib.pyplot as plt


def plot_heatmaps(predictions, ground_truth, seq=0):
    """
    Plots heatmaps of predictions and ground truth for a given sequence index.
    The plot will have 4 columns (2 for predictions and 2 for ground truth) and a number
    of rows equal to the pressure levels. Each heatmap represents one pressure level.

    Parameters:
    - predictions (np.array): Predictions array with dimensions [batch_size, pressure_levels, height, width, channels].
    - ground_truth (np.array): Ground truth array with the same dimensions as predictions.
    - seq (int): Sequence index to plot.
    """
    pressure_levels = predictions.shape[1]
    fig, axs = plt.subplots(pressure_levels, 4, figsize=(8, 20))

    # Set titles for the columns
    column_titles = ['Pred u10', 'Pred v10', 'GT u10', 'GT v10']
    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col)

    for i in range(pressure_levels):
        # Prediction u10
        axs[i, 0].imshow(predictions[seq, i, :, :, 0],
                         cmap='hot', interpolation='nearest')
        axs[i, 0].axis('off')

        # Prediction v10
        axs[i, 1].imshow(predictions[seq, i, :, :, 1],
                         cmap='hot', interpolation='nearest')
        axs[i, 1].axis('off')

        # Ground u10
        axs[i, 2].imshow(ground_truth[seq, i, :, :, 0],
                         cmap='hot', interpolation='nearest')
        axs[i, 2].axis('off')

        # Ground v10
        axs[i, 3].imshow(ground_truth[seq, i, :, :, 1],
                         cmap='hot', interpolation='nearest')
        axs[i, 3].axis('off')

        # Add row labels to the left of the first column, rotated 90 degrees
        axs[i, 0].text(-0.05, 0.5, f'Pressure Level {i+1}', va='center',
                       ha='right', rotation=90, transform=axs[i, 0].transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()
