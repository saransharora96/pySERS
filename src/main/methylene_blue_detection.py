# Notes: 1. Tried filtering spectra, but it didn't have any significant benefit. Decided that simple is better.

from src.utils import raman_plotting_utils as rp, raman_data_processing_utils as rd
import time
from src.install_modules import upgrade_pip, install_packages
from src.utils.data_classes import SensitivityAnalysis, read_dataset
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib.colors import Normalize
from src.config.config_limit_of_detection import (INSTALLATION_NEEDED, INTENSITY_MAP_COLORS, DIGITAL_MAP_COLORS,
                                                  SHOW_DATASET_HEATMAPS, SAVE_DATASET_FIGURES, TRIM_RAMAN_SHIFT_RANGE,
                                                  SHOW_DATASET_REGRESSION_PLOTS, SHOW_DATASET_SPECTRA_PLOTS,
                                                  STD_MULTIPLE, SHOW_MAP_HEATMAP, NOISE_INTENSITY_RANGE,
                                                  CHARACTERISTIC_INTENSITY_RANGE, BACKGROUND_INTENSITY_RANGE)


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
    concentration = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    log_concentration = [np.log10(c) for c in concentration[:len(raman_scans)]]
    hit_count = [scan.summed_intensity for scan in raman_scans][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot in the loop
    for j, scan in enumerate(raman_scans):
        ax.scatter(log_concentration[j], hit_count[j], s=200, label=f"{scan.subdir_name}", color='blue')

    # Monotonic Spline Fit
    try:
        spline = PchipInterpolator(log_concentration, hit_count)
        trendline_x = np.linspace(min(log_concentration), max(log_concentration), 500)
        trendline_y = spline(trendline_x)

        # Plot the spline
        ax.plot(trendline_x, trendline_y, 'r--', linewidth=2)
    except ValueError as e:
        print(f"Spline fitting error: {e}")

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/summed_intensity_regression_plot.pdf", format='pdf')

    if SHOW_DATASET_SPECTRA_PLOTS is True:
        plt.show()


def mean_spectra_plot_entire_dataset(raman_scans):

    shift = [20000, 4000, 1250, 500, 200, 0]
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    for j, scan in enumerate(raman_scans):
        if len(scan.spectra_for_mean_plot) > 5:
            mean_spectrum = np.mean(scan.spectra_for_mean_plot, axis=0)
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
        r"\limit_of_detection"
    )

    intensity_color_map = rp.custom_color_map(INTENSITY_MAP_COLORS)
    binary_color_map = ListedColormap(DIGITAL_MAP_COLORS)

    raman_maps = read_dataset(dataset_location, SensitivityAnalysis)
    print("")

    # data pre-processing before sensitivity analysis
    for raman_map in raman_maps:
        print(f"Data pre-processing: {raman_map.subdir_name}")

        # Smoothen, trim and baseline correct the spectra
        smooth_spectra = rd.smooth_spectra(raman_map.spectra, window_length=5, poly_order=3)
        trimmed_spectra, trimmed_raman_shift = rd.trim_spectra_by_raman_shift_range(smooth_spectra,
                                                                                    raman_map.raman_shift,
                                                                                    TRIM_RAMAN_SHIFT_RANGE[0],
                                                                                    None)
        baseline_corrected_spectra = rd.baseline_correction(trimmed_spectra, trimmed_raman_shift, poly_order=3, num_std=0.7)
        retrimmed_spectra, retrimmed_raman_shift = rd.trim_spectra_by_raman_shift_range(baseline_corrected_spectra,
                                                                                        trimmed_raman_shift,
                                                                                        None,
                                                                                        TRIM_RAMAN_SHIFT_RANGE[1])

        raman_map.spectra = retrimmed_spectra
        raman_map.raman_shift = retrimmed_raman_shift

        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # rp.plot_random_spectra(raman_map.spectra, raman_map.raman_shift, ax=ax, num_spectra_to_plot=25, edge_alpha=1)
        # plt.show()

        # Circular crop to remove edge effects and create a mean plot
        size = np.int64(np.ceil(np.sqrt(raman_map.spectra.shape[0])))
        ones_matrix = np.ones([size, size], dtype=int)
        circular_crop_matrix, _ = rd.circular_crop_square_matrix(ones_matrix, percentage=75)
        spectra_circular_crop = rd.filter_spectra_from_binary_mapping(retrimmed_spectra, circular_crop_matrix)
        raman_map.spectra_for_mean_plot = spectra_circular_crop


    for raman_map in raman_maps:
        raman_map.perform_sensitivity_analysis(intensity_color_map,
                                               CHARACTERISTIC_INTENSITY_RANGE,
                                               BACKGROUND_INTENSITY_RANGE,
                                               NOISE_INTENSITY_RANGE,
                                               STD_MULTIPLE,
                                               SHOW_MAP_HEATMAP)

    norm = NonlinearNorm(vmin=0, vmax=10000)

    heatmap_plot_pooled_dataset(raman_maps, colormap=intensity_color_map, norm=norm)
    mean_spectra_plot_entire_dataset(raman_maps)
    digital_plot_pooled_dataset(raman_maps, colormap=binary_color_map, vmin=0, vmax=1)
    regression_plot(raman_maps)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    "Tasks for the rest of the week"
    # TODO: Make note on pybaselines
    # TODO: Make note on why standardization doesn't work (with plot)
    # TODO: Apply baseline correction to reproducibility as well?
    # TODO: Complete literature review for digital SERS
