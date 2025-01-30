from utils import raman_plotting_utils as rp, raman_data_processing_utils as rd
import time
from install_modules import upgrade_pip, install_packages
from utils.data_classes import SensitivityAnalysis, read_dataset
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import numpy as np
from scipy.optimize import curve_fit
from config.config_pesticide_detection_thiabendazole import (INSTALLATION_NEEDED, INTENSITY_MAP_COLORS, DIGITAL_MAP_COLORS,
                                                                 SHOW_DATASET_HEATMAPS, SAVE_DATASET_FIGURES,
                                                                 SHOW_DATASET_REGRESSION_PLOTS, SHOW_DATASET_SPECTRA_PLOTS,
                                                                 STD_MULTIPLE, SHOW_MAP_HEATMAP,
                                                                 CHARACTERISTIC_INTENSITY_RANGE, BACKGROUND_INTENSITY_RANGE,
                                                                 NOISE_INTENSITY_RANGE)


# Define a normalization that emphasizes lower values
class NonlinearNorm(Normalize):
    def __call__(self, value, clip=None):
        # Set all negative values to 0
        value = np.maximum(value, 0)

        # Apply square root transformation for non-linear scaling
        normalized = np.sqrt(value / self.vmax)
        return np.clip(normalized, 0, 1)


def heatmap_plot_pooled_dataset(raman_scans, colormap='viridis', norm=None):
    n_subplots = len(raman_scans)
    n_cols = math.ceil(math.sqrt(n_subplots))
    n_rows = math.ceil(n_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows))

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for j, scan in enumerate(raman_scans):
        ax = axes[j]  # Get the correct axis for each subplot

        im = ax.imshow(scan.intensity_heatmap, cmap=colormap, norm=norm)

        # Add colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        plt.colorbar(im, cax=cax)

        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        ax.spines[:].set_visible(False)

    for i in range(n_subplots, n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/heatmap.pdf", format='pdf')

    if SHOW_DATASET_HEATMAPS is True:
        plt.show()

    plt.close()


def digital_plot_pooled_dataset(raman_scans, colormap='viridis', vmin=None, vmax=None):
    n_subplots = len(raman_scans)
    n_cols = math.ceil(math.sqrt(n_subplots))
    n_rows = math.ceil(n_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows))

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for j, scan in enumerate(raman_scans):
        ax = axes[j]  # Get the correct axis for each subplot

        # Create the heatmap with pcolormesh
        flipped_map = np.flipud(scan.digitized_map)
        X, Y = np.meshgrid(np.arange(flipped_map.shape[1] + 1), np.arange(flipped_map.shape[0] + 1))
        im = ax.pcolormesh(X, Y, flipped_map, cmap=colormap, vmin=vmin, vmax=vmax, edgecolors='white',
                           linewidth=0.01)
        ax.set_aspect('equal')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.3)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0, 1])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        ax.spines[:].set_visible(False)

    for i in range(n_subplots, n_rows * n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2)

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/digital_map.pdf", format='pdf')

    if SHOW_DATASET_HEATMAPS is True:
        plt.show()

    plt.close()


from scipy.interpolate import PchipInterpolator  # Monotonic Spline Interpolation


def regression_plot(raman_scans):
    # Hit Count in Bulk
    concentration = [1e-1, 1e+0, 1e+1, 1e+2, 1e+3]
    log_concentration = [np.log10(c) for c in concentration[:len(raman_scans)]]
    hit_count = [scan.summed_intensity for scan in raman_scans][::-1]
    log_hit_count = np.log10(hit_count)  # Log-transform the hit counts for fitting

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    for j, scan in enumerate(raman_scans):
        ax.scatter(log_concentration[j], hit_count[j], s=200, label=f"{scan.subdir_name}", color='blue')

    # Set Y-axis to log scale
    ax.set_yscale('log')

    # Monotonic Spline Fit
    try:
        spline = PchipInterpolator(log_concentration, log_hit_count)  # Fit on log-transformed data
        trendline_x = np.linspace(min(log_concentration), max(log_concentration), 500)
        trendline_y_log = spline(trendline_x)
        trendline_y = 10**trendline_y_log  # Transform back to original scale

        # Plot the spline
        ax.plot(trendline_x, trendline_y, 'r--', linewidth=2)
    except ValueError as e:
        print(f"Spline fitting error: {e}")

    # Add labels and legend
    ax.set_xlabel('Log Concentration')
    ax.set_ylabel('Hit Count (Log Scale)')

    # Save and show
    if SHOW_DATASET_REGRESSION_PLOTS:
        plt.savefig(f"{raman_scans[0].base_dir}/hit_count_spline_plot.pdf", format='pdf')

    if SHOW_DATASET_SPECTRA_PLOTS:
        plt.show()


def mean_spectra_plot_entire_dataset(raman_scans):

    shift = [4000, 1000, 350, 125, 0]
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    for j, scan in enumerate(raman_scans):
        if len(scan.spectra) > 5:
            mean_spectrum = np.mean(scan.spectra, axis=0)
            axs.plot(scan.raman_shift, mean_spectrum+((j+1)*shift[j%len(shift)]), label='Mean Spectrum')
            axs.set_title("Mean spectra")
            axs.set_xlabel('Raman Shift (cm^-1)')
            axs.set_ylabel('Intensity (a.u.)')
            axs.yaxis.set_ticks([])

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/mean_spectra.pdf", format='pdf')

    if SHOW_DATASET_SPECTRA_PLOTS is True:
        plt.show()

    plt.close()


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
        r"\pesticide_detection\thiabendazole"
    )

    intensity_color_map = rp.custom_color_map(INTENSITY_MAP_COLORS)
    binary_color_map = ListedColormap(DIGITAL_MAP_COLORS)

    raman_maps = read_dataset(dataset_location, SensitivityAnalysis)
    print("")

    # data pre-processing before sensitivity  (pre-processed in Horiba software)
    for raman_map in raman_maps:
        print(f"Data pre-processing: {raman_map.subdir_name}")

        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # rp.plot_random_spectra(raman_map.spectra, raman_map.raman_shift, ax=ax, num_spectra_to_plot=25, edge_alpha=1)
        # plt.show()

    print("")
    for raman_map in raman_maps:
        raman_map.perform_sensitivity_analysis(intensity_color_map,
                                               CHARACTERISTIC_INTENSITY_RANGE,
                                               BACKGROUND_INTENSITY_RANGE,
                                               NOISE_INTENSITY_RANGE,
                                               STD_MULTIPLE,
                                               SHOW_MAP_HEATMAP)

    # Apply the nonlinear normalization
    norm = NonlinearNorm(vmin=0, vmax=2000)

    heatmap_plot_pooled_dataset(raman_maps, colormap = intensity_color_map, norm=norm)
    digital_plot_pooled_dataset(raman_maps, colormap = binary_color_map, vmin=0, vmax=1)

    mean_spectra_plot_entire_dataset(raman_maps)
    regression_plot(raman_maps)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

