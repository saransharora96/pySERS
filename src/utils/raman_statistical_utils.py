# TODO: All these codes so far assume that the Raman shifts are from the same calibration. I haven't had to deal with
#  interpolation yet. When it comes to that, it should be done on a common base for everything before moving on.

import numpy as np
import matplotlib.pyplot as plt


"""
statistical functions
"""


def percentage_rsd(scalar_matrix, name='the sample', count_zeros=False):
    """
    Calculates the coefficient of variation for the non-zero values in a given matrix, presented as a percentage.

    Parameters:
    matrix (numpy.ndarray): Input matrix with potential zero values.
    name (str): Name of the matrix for display purposes.

    Returns:
    float: Coefficient of variation for the non-zero values, as a percentage.
    """
    if count_zeros is False:
        non_zero_values = scalar_matrix[np.nonzero(scalar_matrix)]  # Extract non-zero values
        scalar_matrix = non_zero_values
    percentage_rsd = round(np.std(scalar_matrix) / np.mean(scalar_matrix) * 100,2)  # # Calculate the coefficient of variation
    print(f"The coefficient of variation for Raman shift = {name} is {percentage_rsd}%.")

    return percentage_rsd


def percentage_rsd_versus_raman_shift(spectra_map, raman_shift, title='Coefficient of Variation vs. Raman Shift'):
    """
    Plots the coefficient of variation (CV) against raman_shift for a given dataset.

    Parameters:
    data (numpy.ndarray): A 2D array where each row is a spectrum and each column is a raman shift
    """
    # Calculate mean and standard deviation along the columns (for each raman_shift)
    mean = np.mean(spectra_map, axis=0)
    std = np.std(spectra_map, axis=0)

    # Calculate the coefficient of variation (CV)
    rsd_spectra = std / mean

    # Plot CV against raman_shift
    plt.figure(figsize=(10, 6))
    plt.plot(raman_shift, rsd_spectra*100, marker='o')
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Coefficient of Variation (%)')
    plt.title(title)