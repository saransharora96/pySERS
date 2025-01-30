"""
Note 1: Prominence can used instead of peak intensity to correct for fluorescence background.
Lieberfit is not used as it was affecting the quality of the cov_results and adding additional artifacts.
Prominence works better in this case. However, absolute value without the use of either Lieberfit or prominence works
much better as some peaks have higher variability compared to others in the same spectra with the peak prominence.
"""
import time
from config.config_reproducibility import (INSTALLATION_NEEDED, MAP_COLORS, SAVE_DATASET_CSV, TARGET_RAMAN_SHIFTS,
                                               SAVE_DATASET_FIGURES, SHOW_DATASET_HEATMAPS, SHOW_DATASET_SPECTRA_PLOTS,
                                               SHOW_DATASET_CONTOUR_PLOTS)
from install_modules import upgrade_pip, install_packages
from utils import raman_plotting_utils as rp, raman_statistical_utils as rs
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from src.utils.data_classes import ReproducibilityAnalysis, read_dataset


def prepare_cov_results(raman_scans, scan_index_size, target_shifts):
    """Prepare cov_results for coefficient of variation for each target_shift, each batch, and the entire dataset."""
    batch_data = {}
    target_shift_data = {target_shift: np.empty((0, scan_index_size)) for target_shift in target_shifts}  # Data per target_shift

    # Collect data
    for scan in raman_scans:
        batch_name = " ".join(scan.subdir_name.split()[:2])  # Extract Batch name
        if batch_name not in batch_data:
            batch_data[batch_name] = {target_shift: np.empty((0, scan_index_size)) for target_shift in target_shifts}

        for target_shift, intensities in scan.intensities_square_dict.items():
            if intensities is not None:
                batch_data[batch_name][target_shift] = np.concatenate((batch_data[batch_name][target_shift], intensities), axis=0)
                target_shift_data[target_shift] = np.concatenate((target_shift_data[target_shift], intensities), axis=0)

    # Calculate coefficients of variation
    results = {"shifts": {target_shift: {} for target_shift in target_shifts}, "batches": {}, "overall": {}}

    # COV for each target_shift in each batch and overall
    for batch_name, data in batch_data.items():
        print(f"\033[93m\nResults for {batch_name}:\033[0m")
        if batch_name not in results["batches"]:
            results["batches"][batch_name] = {}

        for target_shift in target_shifts:
            cov_raman_shift_batch = rs.percentage_rsd(data[target_shift], target_shift)
            results["batches"][batch_name][target_shift] = cov_raman_shift_batch

    print(f"\033[92m\nResults for overall dataset:\033[0m")
    for target_shift in target_shifts:
        cov_overall = rs.percentage_rsd(target_shift_data[target_shift], target_shift)
        max_value = np.max(target_shift_data[target_shift])  # Calculate the maximum value for the target_shift using NumPy
        # Filter out zero values and then find the minimum
        non_zero_values = target_shift_data[target_shift][target_shift_data[target_shift] != 0]
        min_value = np.min(
            non_zero_values) if non_zero_values.size > 0 else None  # Handle case where all values might be zero
        results["shifts"][target_shift]["overall"] = cov_overall
        results["shifts"][target_shift]["max"] = max_value  # Store the maximum value
        results["shifts"][target_shift]["min"] = min_value  # Store the minimum non-zero value

    return results


def save_results_to_csv(cov_results, raman_scans, dataset_location):
    filename = os.path.join(dataset_location, "reproducibility_results.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Scope", "Batch/Chip/Shift", "Coefficient of Variation"])

        # Save chip-specific cov_results
        for raman_scan in raman_scans:
            for key, value in raman_scan.cov_dict.items():
                writer.writerow([raman_scan.subdir_name, key, value])

        # Save batch-specific cov_results
        for batch, shifts in cov_results["batches"].items():
            for shift, cov in shifts.items():
                writer.writerow([batch, shift, cov])

        # Save overall cov_results for each shift
        for shift, data in cov_results["shifts"].items():
            if "overall" in data:
                writer.writerow(["Overall", shift, data["overall"]])


def contour_plot_pooled_dataset(raman_scans, num_spectra, colormap='viridis'):
    """
    Pools all spectra from a list of Raman scan objects, shuffles them, and extracts a specified number of random
    spectra. Then plots a contour plot of the selected spectra.

    Parameters:
    - raman_scans (list): A list of objects, each with a 'filtered_spectra' and 'raman_shift' attribute.
    - num_spectra (int): The number of random spectra to extract.

    Returns:
    - numpy.ndarray: A 2D array containing the randomly selected spectra.
    """
    all_spectra = []
    all_raman_shifts = []

    for scan in raman_scans:
        spectra = scan.filtered_spectra
        raman_shift = scan.raman_shift

        # Convert spectra to numpy array if it's not already one
        if not isinstance(spectra, np.ndarray):
            spectra = np.array(spectra)

        # Convert raman_shift to numpy array if it's not already one
        if not isinstance(raman_shift, np.ndarray):
            raman_shift = np.array(raman_shift)

        # Append spectra and expand raman_shift to match the number of spectra
        all_spectra.append(spectra)
        for _ in range(spectra.shape[0]):
            all_raman_shifts.append(raman_shift)  # Append raman_shift repeated for each spectrum

    # Concatenate all data into one array
    combined_spectra = np.vstack(all_spectra)
    combined_raman_shifts = np.vstack(all_raman_shifts)

    # Generate indices and shuffle them
    indices = np.arange(combined_spectra.shape[0])
    np.random.shuffle(indices)

    # Check if the requested number of spectra is available
    if combined_spectra.shape[0] < num_spectra:
        raise ValueError("Requested number of spectra exceeds the available number.")

    # Select the first 'num_spectra' indices for spectra and raman shifts
    selected_spectra = combined_spectra[indices[:num_spectra]]
    selected_raman_shifts = combined_raman_shifts[indices[:num_spectra]]

    # Plotting the selected spectra
    rp.contour_plot(selected_spectra, selected_raman_shifts, colormap=colormap, save_location=raman_scans[0].base_dir)

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/contour_plot.pdf", format='pdf')

    if SHOW_DATASET_CONTOUR_PLOTS is True:
        plt.show()

    plt.close()

def mean_spectra_plot_pooled_dataset(raman_scans, title='Pooled Mean Spectrum with Standard Deviation'):
    """
    Pools spectra from a list of Raman scan objects, computes the mean spectrum, and plots it along with its
    standard deviation.

    Parameters:
    - raman_scans (list): A list of objects, each with a 'filtered_spectra' and 'raman_shift' attribute.
    - ax (matplotlib axis, optional): Axis on which to plot the spectra.
    - title (str): Title for the plot.
    """

    pooled_spectra = []
    raman_shift = None

    # Pool all spectra and assume raman_shift is the same for all scans
    for scan in raman_scans:
        if not isinstance(scan.filtered_spectra, np.ndarray):
            scan_spectra = np.array(scan.filtered_spectra)
        else:
            scan_spectra = scan.filtered_spectra

        pooled_spectra.append(scan_spectra)

        # Assume Raman shift is consistent across all scans
        if raman_shift is None:
            raman_shift = scan.raman_shift

    # Stack all spectra into a single numpy array
    pooled_spectra = np.vstack(pooled_spectra)

    # Compute the mean and standard deviation
    mean_spectrum = np.mean(pooled_spectra, axis=0)
    std_spectrum = np.std(pooled_spectra, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot mean and standard deviation
    ax.plot(raman_shift, mean_spectrum, label='Mean Spectrum')
    ax.fill_between(raman_shift, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                    color='gray', alpha=0.5, label='Standard Deviation')
    ax.set_title(title)
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity')
    ax.legend()

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/mean_spectra.pdf", format='pdf')

    if SHOW_DATASET_SPECTRA_PLOTS:
        plt.show()

    plt.close()


def heatmap_plot_pooled_dataset(raman_scans, colormap='viridis'):
    """
    Plots a figure with subplots where each column corresponds to a different RamanScan object
    and each row corresponds to a different target shift value.

    Parameters:
    raman_scans (list): A list of RamanScan objects. Each object has a property `intensities_square_dict`
                        which is a dictionary with target shift values as keys and square matrices as values.
    colormap (str): The colormap to use for the heatmaps.
    save_figures (bool): If True, save the figure as a PDF in the base directory of the first RamanScan object.
    show_heatmaps (bool): If True, display the heatmap figures.
    """

    # Determine the unique target shift values across all objects
    target_shifts = sorted(list(raman_scans[0].intensities_square_dict.keys()))

    # Calculate global min and max for each target shift
    global_min_max = {}
    for shift in target_shifts:
        all_matrices = [scan.intensities_square_dict[shift] for scan in raman_scans]
        non_zero_values = np.hstack([matrix[matrix > 0] for matrix in all_matrices])  # Only non-zero values
        min_val = non_zero_values.min() if len(non_zero_values) > 0 else 0  # Handle case with all zeroes
        max_val = non_zero_values.max() if len(non_zero_values) > 0 else 1  # Prevent division by zero
        global_min_max[shift] = (min_val, max_val)

    # Create subplots
    n_objects = len(raman_scans)
    n_shifts = len(target_shifts)

    fig, axes = plt.subplots(n_shifts, n_objects, figsize=(3*n_objects, 4*n_shifts))

    # Plotting the matrices
    for j, scan in enumerate(raman_scans):
        for i, shift in enumerate(target_shifts):
            ax = axes[i, j] if n_shifts > 1 else axes[j]  # Handle case with only one target shift
            matrix = scan.intensities_square_dict[shift]

            # Get the global min and max for this target shift
            vmin, vmax = global_min_max[shift]

            # Plot the matrix
            im = ax.imshow(matrix, vmin=0.8*vmin, vmax=1.2*vmax, cmap=colormap)

            # Add colorbar for each subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            ax.set_title(f"Raman Shift: {shift} cm-1")
            ax.axis('off')  # Optional: turn off the axis labels

            # Add scale bar
            scale_bar = AnchoredSizeBar(ax.transData, 10, '0.5 mm', 'lower right', pad=0, color='black',
                                        frameon=False, size_vertical=2)
            ax.add_artist(scale_bar)

    plt.tight_layout()

    if SAVE_DATASET_FIGURES is True:
        plt.savefig(f"{raman_scans[0].base_dir}/heatmap.pdf", format='pdf')

    if SHOW_DATASET_HEATMAPS is True:
        plt.show()

    plt.close()


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
        r"\reproducibility"
    )

    raman_maps = read_dataset(dataset_location, ReproducibilityAnalysis)
    color_map = rp.custom_color_map(MAP_COLORS)

    for raman_map in raman_maps:
        raman_map.perform_reproducibility_analysis(color_map)

    map_index_size = np.int64(np.sqrt(raman_maps[0].spectra.shape[0]))  # scan size for a square size
    cov_data = prepare_cov_results(raman_maps, map_index_size, TARGET_RAMAN_SHIFTS)  # Prepare overall cov_results

    contour_plot_pooled_dataset(raman_maps, num_spectra=1000, colormap=color_map)
    mean_spectra_plot_pooled_dataset(raman_maps)
    heatmap_plot_pooled_dataset(raman_maps, colormap=color_map)

    if SAVE_DATASET_CSV is True:
        save_results_to_csv(cov_data, raman_maps, dataset_location)  # Save cov_results

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")