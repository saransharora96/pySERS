from src.utils import raman_data_processing_utils as rd, raman_plotting_utils as rp, \
    raman_statistical_utils as rs
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from src.config.config_reproducibility import INSTALLATION_NEEDED, SHOW_MAP_SPECTRA_PLOTS, SHOW_MAP_HEATMAP, SHOW_MAP_PEAK_PLOTS, SHOW_MAP_RSD_PLOTS, \
    REPORT_MAP_PERCENTAGE_RSD, SAVE_MAP_FIGURES, PROMINENCE_THRESHOLD, Z_THRESHOLD, BLOB_SIZE_THRESHOLD, PEAK_SHIFT_TOLERANCE, TARGET_RAMAN_SHIFTS,  \
    COLOR_BAR_RANGE, MAP_COLORS

if INSTALLATION_NEEDED:
    from src.install_modules import upgrade_pip, install_packages


def recalculate_intensities(spectra, raman_shift, index_mapping, square_size):
    """
    Processes intensities for given Raman shifts and returns a dictionary of intensities matrices and their index mappings.

    :param square_size:
    :param index_mapping:
    :param spectra: Array or list of smoothed spectra.
    :param raman_shift: Array or list corresponding to the Raman shift values.
    """
    intensities_square_dict = {}
    for target_shift in TARGET_RAMAN_SHIFTS:
        _, peak_intensities, _, _ = rd.find_peaks_closest_to_target_raman_shift(spectra, raman_shift, target_shift, PROMINENCE_THRESHOLD, PEAK_SHIFT_TOLERANCE)

        # Reshape the peak intensities to a square matrix and store it in the dictionary
        intensities_square_dict[target_shift] = \
            rd.reshape_to_square_matrix_with_filtered_indices(peak_intensities, index_mapping, square_size)

    return intensities_square_dict


def plot_heatmap_results(intensity, subdir_path, color_map='hot', title='Heatmap Results'):
    """
    Plots the Rayleigh and Raman intensity as heatmaps for all Raman shifts of interest in a flexible grid.

    Parameters:
    intensity (dict of numpy.ndarray): All the intensity matrices at different Raman shifts of interest.
    subdir_path (str): Directory path for saving the figures.
    intensity_color_map (str): Color map for the heatmaps.
    title (str): Title for the entire figure.
    """
    n_items = len(intensity)  # Total number of items to plot
    n_cols = min(n_items, 3)  # Determine the number of columns, up to a maximum of 3
    n_rows = (n_items + n_cols - 1) // n_cols  # Calculate the number of rows needed

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 8), squeeze=False)  # Increase figsize for more space
    ax_flat = ax.flatten()  # Flatten the array of axes for easier handling
    for i, (key, value) in enumerate(intensity.items()):
        if int(key) in TARGET_RAMAN_SHIFTS and SAVE_MAP_FIGURES is True:
            index = TARGET_RAMAN_SHIFTS.index(int(key))
            clim = COLOR_BAR_RANGE[index]
        else:
            clim = (None, None)  # default clim if key not in TARGET_RAMAN_SHIFTS

        rp.plot_heatmap(value, color_map=color_map, ax=ax_flat[i])
        ax_flat[i].set_title(f'Raman Shift: {key} cm-1')

        scale_bar = AnchoredSizeBar(ax_flat[i].transData, 10, '0.5 mm', 'lower right', pad=0, color='black',
                                   frameon=False, size_vertical=2)
        ax_flat[i].add_artist(scale_bar)

    # Hide or remove any remaining axes if there are fewer plots than subplots
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)  # Hide empty subplots

    fig.suptitle(title, fontsize=16, y=0.98)  # Adjust the y-position of the title to create more room
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to give more space around the edges
    fig.subplots_adjust(hspace=0.1, wspace=0.2)  # Increase space between subplots

    if SHOW_MAP_HEATMAP: plt.show()  # Show the figure if the flag is set to True
    fig.savefig(f"{subdir_path}/heatmap_results_{title}.pdf", format='pdf') if SAVE_MAP_FIGURES else None  # Save if the flag is True
    plt.close(fig)  # Explicitly close the figure to avoid memory leaks


def plot_spectra_results(spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title='Spectra Results'):

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))  # Adjust subplot grid, ensure ax is always 2D
    selected_spectra, _ = rd.select_random_spectra(spectra, num_spectra)
    rp.plot_raw_spectra(selected_spectra, raman_shift, edge_alpha=edge_alpha, ax=ax[0], title=title)
    rp.plot_mean_spectra(spectra, raman_shift, ax=ax[1], title=title)
    fig.tight_layout()
    if SHOW_MAP_SPECTRA_PLOTS: plt.show()
    fig.savefig(f"{subdir_path}/spectra_results_{title}.pdf", format='pdf') if SAVE_MAP_FIGURES else None  # Save if the flag is True
    plt.close(fig)  # Explicitly close the figure to avoid memory leaks


def report_stepwise_percentage_rsd(intensity, title):

    if REPORT_MAP_PERCENTAGE_RSD is True:
        print(f"\033[94m\nResults after {title}\033[0m")
        for target_shift in TARGET_RAMAN_SHIFTS:
            rs.percentage_rsd(intensity[target_shift], target_shift)


def reproducibility_analysis_for_each_map(spectra, raman_shift, subdir_path, color_map='viridis'):
    """
    This function performs the reproducibility analysis for one map of Raman spectra.
    :param subdir_path: path where the file is stored
    :param color_map: color scheme for heatmaps
    :param spectra: numpy array of Raman spectra
    :param raman_shift: numpy array of raman shifts for those spectra
    :return: saves output figures as .pdf files and returns statistical parameters and processed spectra and maps
    """

    """
    Scale and smoothen the spectra
    """
    scaled_spectra = rd.min_max_normalize_entire_dataset(spectra)
    smoothed_spectra = rd.smooth_spectra(scaled_spectra)

    """
    Find the peak intensity values at different Raman shifts 
    """
    intensities_square_dict = {}
    for target_shift in TARGET_RAMAN_SHIFTS:

        peak_indices, peak_intensities, _, _ = \
            rd.find_peaks_closest_to_target_raman_shift(smoothed_spectra, raman_shift, target_shift,
                                                                PROMINENCE_THRESHOLD, PEAK_SHIFT_TOLERANCE)

        if SHOW_MAP_PEAK_PLOTS is True:
            rp.plot_random_spectra_with_peaks(spectra, peak_indices, raman_shift, num_spectra_to_plot=12,
                                                       edge_alpha=1)
            plt.subplots_adjust(top=0.85)
            plt.suptitle(f'Target Raman Shift: {target_shift}', fontsize=16, y=0.98)
            plt.show()
            plt.close()  # Explicitly close the figure to avoid memory leaks

        intensities_square_dict[target_shift], index_mapping = rd.reshape_to_square_matrix(peak_intensities)
        index_mapping, index_mapping_binary = rd.index_mapping_update(index_mapping, indices=[])

    title = 'Raw Data'
    plot_spectra_results(smoothed_spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title=title)
    plot_heatmap_results(intensities_square_dict, subdir_path, color_map=color_map, title=title)

    """
    Filter the spectra based on whether the Rayleigh and Raman peaks are found or not 
    """
    for target_shift in TARGET_RAMAN_SHIFTS:
        _, peak_intensities, peak_raman_shift, peak_prominences  = \
            rd.find_peaks_closest_to_target_raman_shift(smoothed_spectra, raman_shift, target_shift, PROMINENCE_THRESHOLD, PEAK_SHIFT_TOLERANCE)
        peak_raman_shift_square, _ = rd.reshape_to_square_matrix(peak_raman_shift)
        filter_indices = rd.find_indexes_where_matrix_value_is_not_at_target_value(peak_raman_shift_square, target=target_shift,
                                                                                   tolerance=PEAK_SHIFT_TOLERANCE)
        index_mapping, _ = rd.index_mapping_update(index_mapping, filter_indices)

        filtered_spectra, spectra_indices = rd.filter_from_index_mapping(smoothed_spectra, index_mapping)
        filtered_spectra = rd.min_max_normalize_entire_dataset(filtered_spectra)

        if SHOW_MAP_PEAK_PLOTS is True:
            rp.plot_filtered_closest_peaks_in_map(peak_raman_shift, index_mapping, peak_intensities,
                                                           title=f'Filtered Closest Peaks for Target Raman Shift: {target_shift}')
            plt.show()
            plt.close()  # Explicitly close the figure to avoid memory leaks

    square_size = np.int64(np.sqrt(spectra.shape[0]))  # scan size for a square size
    intensities_square_dict = recalculate_intensities(filtered_spectra, raman_shift, index_mapping, square_size)

    title = 'filtering spectra with absent peaks'
    report_stepwise_percentage_rsd(intensities_square_dict, title)
    plot_spectra_results(filtered_spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title=title)
    plot_heatmap_results(intensities_square_dict, subdir_path, color_map=color_map, title=title)
    if SHOW_MAP_RSD_PLOTS is True:
        rs.percentage_rsd_versus_raman_shift(filtered_spectra, raman_shift, title)
        plt.show()
        plt.close()  # Explicitly close the figure to avoid memory leaks

    """
    Normalize the spectra
    """
    standardized_spectra = rd.standardization(filtered_spectra)
    standardized_spectra = rd.min_max_normalize_entire_dataset(standardized_spectra)
    intensities_square_dict = recalculate_intensities(standardized_spectra, raman_shift, index_mapping, square_size)

    title = 'after standardization'
    report_stepwise_percentage_rsd(intensities_square_dict, title)
    plot_spectra_results(standardized_spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title=title)
    plot_heatmap_results(intensities_square_dict, subdir_path, color_map=color_map, title=title)
    if SHOW_MAP_RSD_PLOTS is True:
        rs.percentage_rsd_versus_raman_shift(standardized_spectra, raman_shift, title)
        plt.show()
        plt.close()
    """
    Filter the spectra based on outliers (z score)
    """
    filtered_spectra, filtered_indices = rd.remove_outliers_from_spectra(standardized_spectra, spectra_indices,
                                                                         square_size, threshold=Z_THRESHOLD)

    index_mapping, index_mapping_binary = rd.index_mapping_update(index_mapping, filtered_indices)
    filtered_spectra = rd.min_max_normalize_entire_dataset(filtered_spectra)
    intensities_square_dict = recalculate_intensities(filtered_spectra, raman_shift, index_mapping, square_size)

    title = 'filtering based on outliers in Z-score'
    report_stepwise_percentage_rsd(intensities_square_dict, title)
    plot_spectra_results(filtered_spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title=title)
    plot_heatmap_results(intensities_square_dict, subdir_path, color_map=color_map, title=title)
    if SHOW_MAP_RSD_PLOTS is True:
        rs.percentage_rsd_versus_raman_shift(filtered_spectra, raman_shift, title)
        plt.show()
        plt.close()

    """
    Filter the spectra based on spatial outliers (small blobs)
    """
    _, removed_points = rd.remove_small_blobs(intensities_square_dict[TARGET_RAMAN_SHIFTS[0]],BLOB_SIZE_THRESHOLD)
    index_mapping, index_mapping_binary = rd.index_mapping_update(index_mapping, removed_points)
    filtered_spectra, spectra_indices = rd.filter_from_index_mapping(smoothed_spectra, index_mapping)
    filtered_spectra = rd.standardization(filtered_spectra)
    filtered_spectra = rd.min_max_normalize_entire_dataset(filtered_spectra)
    intensities_square_dict = recalculate_intensities(filtered_spectra, raman_shift, index_mapping, square_size)

    title = 'filtering based on spatial outliers (small blobs)'
    report_stepwise_percentage_rsd(intensities_square_dict, title)
    plot_spectra_results(filtered_spectra, raman_shift, subdir_path, num_spectra=100, edge_alpha=0.25, title=title)
    plot_heatmap_results(intensities_square_dict, subdir_path, color_map=color_map, title=title)
    if SHOW_MAP_RSD_PLOTS is True:
        rs.percentage_rsd_versus_raman_shift(filtered_spectra, raman_shift, title)
        plt.show()
        plt.close()

    return filtered_spectra, intensities_square_dict, index_mapping


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive - Johns Hopkins\Desktop\Johns Hopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
        r"\reproducibility\Batch 1 Chip 1"
    )

    color_map = rp.custom_color_map(MAP_COLORS)

    for file in os.listdir(dataset_location):
        if file.endswith('.txt'):
            raman_shift, spectra = rd.read_horiba_raman_txt_file(os.path.join(dataset_location, file))
            filtered_spectra, intensities_square_dict, index_mapping = \
                reproducibility_analysis_for_each_map(spectra, raman_shift, dataset_location, color_map=color_map)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")