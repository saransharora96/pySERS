from src.utils import raman_plotting_utils as rp, raman_data_processing_utils as rd
import time
from src.install_modules import upgrade_pip, install_packages
from src.utils.data_classes import SensitivityAnalysis, read_dataset
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import numpy as np
from src.config.config_pesticide_detection_multiplexing import (INSTALLATION_NEEDED,
                                                          INTENSITY_MAP_COLORS_THIRAM, DIGITAL_MAP_COLORS_THIRAM,
                                                          INTENSITY_MAP_COLORS_THIABENDAZOLE, DIGITAL_MAP_COLORS_THIABENDAZOLE,
                                                          SHOW_DATASET_HEATMAPS, SAVE_DATASET_FIGURES,
                                                          SHOW_DATASET_SPECTRA_PLOTS,
                                                          STD_MULTIPLE, SHOW_MAP_HEATMAP,
                                                          CHARACTERISTIC_INTENSITY_RANGE_THIRAM, BACKGROUND_INTENSITY_RANGE_THIRAM,
                                                          CHARACTERISTIC_INTENSITY_RANGE_THIABENDAZOLE,
                                                          BACKGROUND_INTENSITY_RANGE_THIABENDAZOLE,
                                                          NOISE_INTENSITY_RANGE)


# Define a normalization that emphasizes lower values
class NonlinearNorm(Normalize):
    def __call__(self, value, clip=None):
        # Set all negative values to 0
        value = np.maximum(value, 0)

        # Apply square root transformation for non-linear scaling
        normalized = np.sqrt(value / self.vmax)
        return np.clip(normalized, 0, 1)


def heatmap_plot_pooled_dataset(raman_scans, colormap='viridis', norm=None,title=""):
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
        plt.savefig(f"{raman_scans[0].base_dir}/heatmap_{title}.pdf", format='pdf')

    if SHOW_DATASET_HEATMAPS is True:
        plt.show()

    plt.close()


def digital_plot_pooled_dataset(raman_scans, colormap='viridis', vmin=None, vmax=None,title=""):
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
        plt.savefig(f"{raman_scans[0].base_dir}/digital_map_{title}.pdf", format='pdf')

    if SHOW_DATASET_HEATMAPS is True:
        plt.show()

    plt.close()


def plot_combined_heatmap(raman_scans,heatmap1, heatmap2):
    # Combine the two heatmaps into a single array with unique values for each condition
    combined = heatmap1 * 2 + heatmap2  # Unique values: 0, 1, 2, 3

    # Define the colormap with the desired colors
    cmap = ListedColormap(['#FFFFFF', '#F69D6E', "#A1D99B","#594525"])

    # Create the plot
    fig, ax = plt.subplots()
    cax = ax.imshow(combined, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)

    # Add a legend for clarity
    colorbar = plt.colorbar(cax, ticks=[0, 1, 2, 3])
    colorbar.ax.set_yticklabels(['Both 0', 'Second 1, First 0', 'First 1, Second 0', 'Both 1'])

    ax.set_title("Combined Heatmap")
    plt.axis('off')

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/digital_maps_combined.pdf", format='pdf')

    plt.show()


def mean_spectra_plot_entire_dataset(raman_scans):
    shift = [500, 0]
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))

    for j, scan in enumerate(raman_scans):
        if len(scan.spectra) > 5:
            mean_spectrum = np.mean(scan.spectra, axis=0)
            std_spectrum = np.std(scan.spectra, axis=0)

            # Plot the mean spectrum
            axs.plot(scan.raman_shift, mean_spectrum + ((j + 1) * shift[j % len(shift)]), label=f'Scan {j + 1}')

            # Plot the standard deviation as a shaded area
            axs.fill_between(
                scan.raman_shift,
                mean_spectrum - std_spectrum + ((j + 1) * shift[j % len(shift)]),
                mean_spectrum + std_spectrum + ((j + 1) * shift[j % len(shift)]),
                alpha=0.3, label=f'Std Dev (Scan {j + 1})'
            )

    axs.set_title("Mean Spectra with Standard Deviation")
    axs.set_xlabel('Raman Shift (cm^-1)')
    axs.set_ylabel('Intensity (a.u.)')
    axs.yaxis.set_ticks([])
    axs.legend(loc='upper right', fontsize='small')

    if SAVE_DATASET_FIGURES:
        plt.savefig(f"{raman_scans[0].base_dir}/mean_spectra_with_std.pdf", format='pdf')

    if SHOW_DATASET_SPECTRA_PLOTS:
        plt.show()

    plt.close()


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
        r"\pesticide_detection\multiplexing"
    )

    raman_maps = read_dataset(dataset_location, SensitivityAnalysis)
    print("")

    # data pre-processing before sensitivity  (pre-processed in Horiba software)
    for raman_map in raman_maps:
        print(f"Data pre-processing: {raman_map.subdir_name}")

        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # rp.plot_random_spectra(raman_map.spectra, raman_map.raman_shift, ax=ax, num_spectra_to_plot=25, edge_alpha=1)
        # plt.show()

    # THIRAM SET
    intensity_color_map = rp.custom_color_map(INTENSITY_MAP_COLORS_THIRAM)
    binary_color_map = ListedColormap(DIGITAL_MAP_COLORS_THIRAM)

    print("")
    for raman_map in raman_maps:
        raman_map.perform_sensitivity_analysis(intensity_color_map,
                                               CHARACTERISTIC_INTENSITY_RANGE_THIRAM,
                                               BACKGROUND_INTENSITY_RANGE_THIRAM,
                                               NOISE_INTENSITY_RANGE,
                                               STD_MULTIPLE,
                                               SHOW_MAP_HEATMAP)

    thiram_map = raman_maps[1].digitized_map
    # Apply the nonlinear normalization
    norm = NonlinearNorm(vmin=0, vmax=250)

    heatmap_plot_pooled_dataset(raman_maps, colormap=intensity_color_map, norm=norm,title="thiram")
    digital_plot_pooled_dataset(raman_maps, colormap=binary_color_map, vmin=0, vmax=1,title="thiram")

    # THIABENDAZOLE SET
    intensity_color_map = rp.custom_color_map(INTENSITY_MAP_COLORS_THIABENDAZOLE)
    binary_color_map = ListedColormap(DIGITAL_MAP_COLORS_THIABENDAZOLE)

    print("")
    for raman_map in raman_maps:
        raman_map.perform_sensitivity_analysis(intensity_color_map,
                                               CHARACTERISTIC_INTENSITY_RANGE_THIABENDAZOLE,
                                               BACKGROUND_INTENSITY_RANGE_THIABENDAZOLE,
                                               NOISE_INTENSITY_RANGE,
                                               STD_MULTIPLE,
                                               SHOW_MAP_HEATMAP)

    thiabendazole_map = raman_maps[1].digitized_map
    # Apply the nonlinear normalization
    norm = NonlinearNorm(vmin=0, vmax=250)

    heatmap_plot_pooled_dataset(raman_maps, colormap=intensity_color_map, norm=norm,title="thiabendazole")
    digital_plot_pooled_dataset(raman_maps, colormap=binary_color_map, vmin=0, vmax=1,title="thiabendazole")

    mean_spectra_plot_entire_dataset(raman_maps)
    plot_combined_heatmap(raman_maps,thiram_map, thiabendazole_map)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
