# TODO: All these codes so far assume that the Raman shifts are from the same calibration. I haven't had to deal with
#  interpolation yet. When it comes to that, it should be done on a common base for everything before moving on.

import matplotlib.pyplot as plt
import numpy as np
from utils import raman_data_processing_utils as raman_data
from matplotlib.colors import LinearSegmentedColormap
import random
import inspect


"""
spectra plotting functions:
"""


def plot_raw_spectra(spectra, raman_shift, edge_alpha=1, ax=None, title='Spectra'):
    """
    Plots the given spectra against the Raman shift values. This code doesn't make the figure or show the plot,
    that must be done handled in the main (outer code) that calls this function.

    Parameters:
    spectra (numpy array): The dataset containing spectra to be plotted.
    raman_shift (numpy array): The Raman shift values corresponding to the spectra.
    edge_alpha (float): The alpha blending value for the plot edges. Default is 1.
    """
    if ax is None:
        ax = plt.gca()

    for spectrum in spectra:
        ax.plot(raman_shift, spectrum, 'k-', linewidth=0.5, alpha=edge_alpha)
    ax.set_title(title)
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity')


def plot_random_spectra(spectra, raman_shift, ax=None, num_spectra_to_plot=25, edge_alpha=1):
    """
    Selects and plots a random subset of spectra from the given dataset.

    Parameters:
    spectra (numpy array): The dataset containing spectra.
    raman_shift (numpy array): The Raman shift values corresponding to the spectra.
    num_spectra (int): The number of random spectra to select and plot. Default is 25.
    edge_alpha (float): The alpha blending value for the plot edges. Default is 1.
    """
    random_spectra, _ = raman_data.select_random_spectra(spectra, num_spectra_to_plot)
    plot_raw_spectra(random_spectra, raman_shift, edge_alpha=edge_alpha, ax=ax)


def plot_mean_spectra(spectra, raman_shift, ax=None, title='Mean Spectrum with Standard Deviation', y_lim=None, label=None):
    """
    Plots the mean spectrum with standard deviation shading.
    Parameters:
    spectra (numpy array): The dataset containing spectra to be plotted.
    raman_shift (numpy array): The Raman shift values corresponding to the spectra.
    ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis object.
    title (str): Title of the plot.
    y_lim (tuple): Limits for the y-axis.
    label (str): Label for the mean spectrum.
    """
    if ax is None:
        ax = plt.gca()

    spectra_array = np.array(spectra)
    mean_spectrum = np.mean(spectra_array, axis=0)
    std_spectrum = np.std(spectra_array, axis=0)

    # Plot the mean spectrum with the provided label
    ax.plot(raman_shift, mean_spectrum, label=label)

    # Plot the standard deviation shading without a label
    ax.fill_between(raman_shift, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.3, label=None)

    ax.set_title(title)
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity')

    if y_lim is not None:
        ax.set_ylim(y_lim)


def plot_random_spectra_with_peaks(spectra, peak_indices, raman_shift, num_spectra_to_plot=10, edge_alpha=1):
    """
    Plot random spectra with marked peaks.

    Parameters:
    spectra (numpy.ndarray): Array of spectra (total_spectra x num_points).
    peaks (list of numpy.ndarray): List of arrays, where each array contains the indices of peaks for the corresponding spectrum.
    raman_shift (numpy.ndarray): Raman shift values.
    num_spectra (int): Number of spectra to plot. Default is 10.
    edge_alpha (float): Alpha value for plot transparency. Default is 1.
    """

    _, random_indices = raman_data.select_random_spectra(spectra, num_spectra_to_plot)

    num_cols = 4
    num_rows = (num_spectra_to_plot + num_cols - 1) // num_cols  # Ceiling division

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(3*num_cols, 2*num_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes into a 1D array for easy indexing

    for i, idx in enumerate(random_indices):
        spectrum = spectra[idx]
        peak = peak_indices[idx]
        axs[i].plot(raman_shift, spectrum, 'k-', linewidth=0.5, alpha=edge_alpha)
        axs[i].plot(raman_shift[peak], spectrum[peak], 'ro')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])  # Hide any unused subplots

    plt.tight_layout()


def plot_some_spectra_and_mean_spectra(spectra, raman_shift, edge_alpha=0.25, num_spectra_to_plot=100):
    """
    Plots 1) random spectra and 2) mean spectra of the filtered dataset.
    """

    selected_spectra, _ = raman_data.select_random_spectra(spectra, num_spectra_to_plot)

    fig, (ax1,ax2) = plt.subplots(2,1)
    plot_mean_spectra(spectra, raman_shift, ax=ax2)
    plot_raw_spectra(selected_spectra, raman_shift, edge_alpha=edge_alpha, ax=ax1)
    plt.tight_layout()


def plot_closest_peaks(closest_peak_raman_shift, closest_peak_intensity, title=None):
    """
    Plots the closest peaks on a scatter plot.

    Args:
      closest_peak_raman_shift: Raman shift of the closest peak for all spectra.
      closest_peak_intensity: Intensity of the closest peak for all spectra.
      title: The title to be set for the plot. Default is None.
    """

    plt.scatter(closest_peak_raman_shift, closest_peak_intensity)
    plt.title(title, fontsize=16, y=1.05)  # Adjust y for spacing
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity (a.u.)')


def plot_filtered_closest_peaks_in_map(peak_raman_shifts, index_mapping, peak_intensities, title=None):
    peak_raman_shifts, _ = raman_data.filter_from_index_mapping(peak_raman_shifts, index_mapping)
    peak_intensities, _ = raman_data.filter_from_index_mapping(peak_intensities, index_mapping)
    plot_closest_peaks(peak_raman_shifts, peak_intensities, title)


def plot_lieberfit_results(spectra, raman_shift, fluorescence_backgrounds, corrected_spectra):
    """
    Plots 10 random spectra with their computed fluorescent background and corrected spectrum in subplots.

    raman_shift: 1D array of raman_shift.
    spectra: List of 1D arrays of the acquired spectra.
    fluorescence_backgrounds: List of 1D arrays of the computed fluorescent backgrounds.
    corrected_spectra: List of 1D arrays of the corrected spectra.
    """

    indices = random.sample(range(len(spectra)), 10)

    fig, axs = plt.subplots(10, 2, figsize=(10, 20))

    for i, idx in enumerate(indices):
        # Plot acquired spectrum and computed fluorescent background
        axs[i, 0].plot(raman_shift, spectra[idx], label='Acquired Spectrum')
        axs[i, 0].plot(raman_shift, fluorescence_backgrounds[idx], label='Computed Fluorescent Background')
        # Plot corrected spectrum
        axs[i, 1].plot(raman_shift, corrected_spectra[idx])

    plt.tight_layout()


def contour_plot(spectra, raman_shift, colormap='viridis', shift_start=50, save_location=None):
    """
    Creates a Raman contour plot starting from a specified Raman shift value.

    Parameters:
    spectra (list of lists or np.array): Each row/entry is a Raman spectrum.
    raman_shift (list or np.array): Array of raman_shift.
    colormap (str): The colormap to use for the plot. Default is 'viridis'.
    shift_start (int): The starting value of Raman shift to plot. Default is 25.
    """

    # Convert to numpy array if the input is a list of lists
    if isinstance(spectra, list):
        spectra = np.array(spectra)
    if isinstance(raman_shift, list):
        raman_shift = np.array(raman_shift)

    # Ensure raman_shift is a 2D array of the same shape as spectra
    if raman_shift.ndim == 1:
        raman_shift = np.tile(raman_shift, (spectra.shape[0], 1))

    # Find the index where raman_shift values are greater than or equal to shift_start
    start_index = np.argmax(raman_shift[0] >= shift_start)

    # Slice the raman_shift and spectra arrays from the start_index
    raman_shift = raman_shift[:, start_index:]
    spectra = spectra[:, start_index:]

    x = raman_shift[0]
    y = np.arange(spectra.shape[0])

    # Create the contour plot
    fig, ax = plt.subplots()
    contour = ax.contourf(x, y, spectra, cmap=colormap,levels=50, vmin=0, vmax=1)
    cbar = fig.colorbar(contour, ticks=np.linspace(0, 1, 11))
    cbar.set_label('Intensity')
    ax.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax.set_ylabel('Spectrum Index')
    ax.set_title('Raman Contour Plot')
    ax.set_xlim(left=shift_start)

    # Adjust aspect ratio to make the plot look square
    ratio = (x.max() - x.min()) / (y.max() - y.min())
    ax.set_aspect(ratio)

    # Remove borders
    for spine in ax.spines.values():
        spine.set_visible(False)


"""
Heatmap Plotting Functions
"""


def plot_heatmap(matrix, color_map='hot', ax=None, vmin=None, vmax=None, return_im=False):

    if ax is None:
        ax = plt.gca()
    im = ax.imshow(matrix, cmap=color_map, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.spines[:].set_visible(False)  # Remove all spines
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    if return_im:
        return im


def custom_color_map(colors):
    return LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)


import matplotlib.patches as patches
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import Normalize

def plot_confusion_matrix(ax, y_test, y_pred, labels, colors=('#F18521', '#1A99D6')):
    """
    Plots a binary-styled confusion matrix without a colorbar.
    
    Parameters:
    - y_test: Ground truth labels.
    - y_pred: Predicted labels.
    - labels: Unique labels for classes.
    - colors: Tuple of two hex colors for false and true predictions.
    
    Returns:
    - None: Displays the confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cmap = sns.color_palette([colors[1], colors[0]], as_cmap=False)
    alpha = 0.7

    # Plot the confusion matrix using Seaborn heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, linewidths=2, linecolor='white', square=True, cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 16}, alpha=alpha)
    
     # Add a solid split bar on the right
    bar_height = len(cm)  # Relative height of the bar
    bar_width = 0.1  # Relative width of the bar
    left = len(cm) + bar_width  # Position on the right side of the heatmap
    for i, color in enumerate(colors):
        rect = patches.Rectangle(
            (left, i),  # Bottom-left corner
            bar_width,  # Width
            bar_height/2,  # Height
            linewidth=1,
            edgecolor='k',
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(rect)
    
    # Adjust axis limits to include the bar
    ax.set_xlim(0, len(cm)+bar_width+0.15)
    ax.set_ylim(len(cm)+0.1, -0.1)

    # Remove ticks and title but keep axis labels
    ax.tick_params(left=False, bottom=False, labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()


def add_feature_importance_shading(ax, raman_shift, feature_importances, color_hex="#FF0000", max_alpha=0.5):
    """
    Shade the background of the plot based on feature importance with a colorbar.

    Args:
        ax: The axis to apply shading.
        raman_shift: Array of Raman shift values.
        feature_importances: Array of feature importances corresponding to Raman shifts.
        color_hex: Hexadecimal color for the shading.
        max_alpha: Maximum transparency level for the highest importance. Default is 0.5.
    Returns:
        norm: Normalized feature importance used for the colorbar.
        cmap: Colormap instance used for shading.
    """
    # Normalize feature importances to [0, max_alpha]
    scaled_importances = feature_importances / np.max(feature_importances) * max_alpha
    
    # Create a custom colormap using the hex color
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(1, 1, 1, 0), mpl.colors.to_rgba(color_hex, 1)],  # Transparent to opaque gradient
    )
    
    norm = Normalize(vmin=0, vmax=max_alpha)  # Scale to [0, max_alpha]
    
    # Add vertical shading for each feature
    for i in range(len(raman_shift) - 2):
        ax.axvspan(raman_shift[i-1], raman_shift[i + 2], color=cmap(norm(scaled_importances[i])))
    
    return norm, cmap
