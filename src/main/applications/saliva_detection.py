import os
import utils.raman_data_processing_utils as rd
import utils.raman_plotting_utils as rp
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pybaselines import Baseline


def baseline_correction(spectra, raman_shift):
    baseline_fitter = Baseline(x_data=raman_shift)
    baseline_corrected_spectra = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        baseline, _ = baseline_fitter.imodpoly(spectra[i, :], poly_order=3, num_std=0.7)
        baseline_corrected_spectra[i, :] = spectra[i, :] - baseline
    return baseline_corrected_spectra


def read_data_in_folder(dataset_location):
    """
    Reads Raman spectra from text files in a directory, extracts the key from the filename,
    and interpolates the spectra to a common Raman shift axis.

    Parameters:
    - dataset_location (str): Path to the directory containing the Raman spectra text files.
    - common_raman_shift_range (tuple): The range of the common Raman shift axis (default: (1000, 1650)).
    - num_points (int): The number of points for the common Raman shift axis (default: 1000).

    Returns:
    - all_data (dict): Dictionary containing the interpolated spectra data organized by key.
    """
    all_data = defaultdict(list)
    print("\nReading files from directory:", dataset_location, "\n")

    for _, _, files in os.walk(dataset_location):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(dataset_location, file)
                print("Reading file:", file_path)
                # Extract the key from the filename
                key = os.path.splitext(file)[0][:-2]
                raman_shift, spectra = rd.read_horiba_raman_txt_file(file_path)
                all_data[key].append((raman_shift, spectra))

    return all_data


def plot_spectra_grid(all_data, n_cols=3, figsize=(15, 4), h_pad=10, w_pad=5):
    """
    Plots Raman spectra stored in the `all_data` dictionary on a grid of subplots.

    Parameters:
    - all_data (dict): Dictionary where keys are plot titles and values are lists of tuples.
                       Each tuple contains a common Raman shift axis and the corresponding spectra.
    - n_cols (int): Number of columns in the subplot grid (default: 3).
    - figsize (tuple): Size of the figure (default: (15, 4)).
    - h_pad (int): Height padding between subplots (default: 10).
    - w_pad (int): Width padding between subplots (default: 5).

    Returns:
    - None: The function displays the plot grid.
    """
    n_samples = sum(len(spectra_list) for spectra_list in all_data.values())
    n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

    # Initialize figure for subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], n_rows * figsize[1]))
    axes = axes.flatten()

    # Plot individual spectra with titles as keys
    plot_index = 0
    for key, spectra_data in all_data.items():
        for raman_shift, spectra in spectra_data:
            corrected_spectra = baseline_correction(spectra, raman_shift)
            corrected_spectra = rd.smooth_spectra(corrected_spectra, window_length=21, poly_order=5)
            corrected_spectra = rd.min_max_normalize_entire_dataset(corrected_spectra)
            rp.plot_random_spectra(spectra, raman_shift, ax=axes[plot_index], edge_alpha=0.5, num_spectra_to_plot=len(spectra))
            axes[plot_index].set_title(key)
            plot_index += 1

    # Remove any unused subplots
    for j in range(plot_index, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    plt.show()


def plot_pooled_spectra(all_data, n_cols=2, figsize=(12, 3), h_pad=5, window_length=21, poly_order=5):
    """
    Creates and plots pooled spectra for each key in the `all_data` dictionary.

    Parameters:
    - all_data (dict): Dictionary where keys are plot titles and values are lists of tuples.
                       Each tuple contains a common Raman shift axis and the corresponding spectra.
    - n_cols (int): Number of columns in the subplot grid (default: 2).
    - figsize (tuple): Base size of the figure (width, height per row) (default: (15, 2)).
    - h_pad (int): Height padding between subplots (default: 10).
    - window_length (int): Window length for the smoothing function (default: 21).
    - poly_order (int): Polynomial order for the smoothing function (default: 5).

    Returns:
    - None: The function displays the plot grid.
    """
    n_rows = (len(all_data) + n_cols - 1) // n_cols  # Ceiling division

    # Initialize figure for subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], n_rows * figsize[1]))
    axes = axes.flatten()

    if len(all_data) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one key

    for ax, (key, spectra_data) in zip(axes, all_data.items()):
        # Pool all spectra for the current key
        pooled_spectra = np.vstack([spectra for _, spectra in spectra_data])
        raman_shift = spectra_data[0][0]  # Use the Raman shift from the first entry, assuming all are the same
        spectra = baseline_correction(pooled_spectra, raman_shift)
        spectra = rd.smooth_spectra(spectra, window_length=window_length, poly_order=poly_order)
        spectra = rd.min_max_normalize_entire_dataset(spectra)

        mean_spectra = np.mean(spectra, axis=0)
        std_spectra = np.std(spectra, axis=0)

        ax[0].plot(raman_shift, mean_spectra, label='Mean Spectra')
        ax[0].fill_between(raman_shift, mean_spectra - std_spectra, mean_spectra + std_spectra, alpha=0.3,
                        label='Std Deviation')

    plt.tight_layout(h_pad=h_pad)
    plt.show()


dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\biomolecular_sensing\saliva"
)
all_data = read_data_in_folder(dataset_location)
plot_spectra_grid(all_data)
plot_pooled_spectra(all_data, window_length=11, poly_order=5)