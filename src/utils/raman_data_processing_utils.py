# TODO: All these codes so far assume that the Raman shifts are from the same calibration. I haven't had to deal with
#  interpolation yet. When it comes to that, it should be done on a common base for everything before moving on.


import numpy as np
import random
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import label, find_objects
from tqdm import tqdm
from scipy.interpolate import interp1d
from pybaselines import Baseline


"""
file handling functions:
"""


def read_horiba_raman_txt_file(file_path):
    """
    Reads a single .txt file from the specified path and processes its contents as numerical data.
    The function expects the first row of the file to represent the Raman shift and the subsequent rows to contain spectra data.

    Args:
      file_path: str
        The path to the .txt file that contains the data.

    Returns:
      tuple:
        - np.ndarray: A 1D numpy array containing the Raman shift values.
        - np.ndarray: A 2D numpy array where each row corresponds to a spectrum.

    Note:
      The function assumes that the first two columns of the file are not relevant to the Raman shift and spectra data,
      hence they are excluded from the cov_results.
    """

    with open(file_path, 'r') as f:
        data = np.loadtxt(f)

        if data.shape[1] == 2: # Case where the txt file has only 1 spectrum (i.e. data is [C1: Raman shift, C2: spectrum])
            raman_shift = data[:, 0]
            spectra = data[:, 1]

        else: # Case where the txt file has multiple spectra (i.e. data is [R1: Raman shift, R2+:spectrum1, spectrum2, ...])
            data_with_extra_columns_removed = data[:, 2:]
            raman_shift = data_with_extra_columns_removed[0, :]
            spectra = data_with_extra_columns_removed[1:, :]

    return raman_shift, spectra


def interpolate_spectra_to_common_base(spectra, original_shift, common_shift):
    """
    Interpolate 2D spectra to a common Raman shift axis.

    Parameters:
        spectra (numpy.ndarray): 2D array of spectra (shape: [n_spectra, n_shifts]).
        original_shift (numpy.ndarray): Original Raman shift axis (shape: [n_shifts]).
        common_shift (numpy.ndarray): The common Raman shift axis to interpolate to.

    Returns:
        numpy.ndarray: A 2D array of interpolated spectra (shape: [n_spectra, len(common_shift)]).
    """
    interpolated_spectra = []
    for spectrum in spectra:
        # Create an interpolator for each spectrum
        interpolator = interp1d(original_shift, spectrum, kind='linear', bounds_error=False, fill_value=0)
        # Interpolate the spectrum to the common Raman shift axis
        interpolated_spectra.append(interpolator(common_shift))

    return np.array(interpolated_spectra)


import numpy as np
from scipy.interpolate import interp1d


def upsample_raman_spectra(original_axis, spectra, new_density=10):
    """
    Upsamples multiple Raman spectra by increasing the density of the Raman shift axis.

    Parameters:
        original_axis (array-like): Original Raman shift axis (1D array).
        spectra (array-like): 2D array of intensities, where each row is a spectrum.
        new_density (int): Number of additional points per original interval (default: 10).

    Returns:
        new_axis (numpy.ndarray): Upsampled Raman shift axis.
        new_spectra (numpy.ndarray): Interpolated intensities for all spectra on the new axis.
    """
    # Ensure inputs are numpy arrays
    original_axis = np.array(original_axis)
    spectra = np.array(spectra)

    # Validate the input shapes
    if spectra.shape[1] != len(original_axis):
        raise ValueError("The number of columns in 'spectra' must match the length of 'original_axis'.")

    # Calculate the upsampled Raman shift axis
    min_shift = original_axis.min()
    max_shift = original_axis.max()
    num_points = len(original_axis) + (len(original_axis) - 1) * new_density
    new_axis = np.linspace(min_shift, max_shift, num_points)

    # Interpolate each spectrum
    new_spectra = np.zeros((spectra.shape[0], len(new_axis)))
    for i, intensity in enumerate(spectra):
        interpolation_function = interp1d(original_axis, intensity, kind='cubic')
        new_spectra[i, :] = interpolation_function(new_axis)

    return new_axis, new_spectra


"""
preprocessing functions
"""


def select_random_spectra(spectra, num_spectra_to_plot=25):
    """
    Selects a random subset of spectra from the given dataset.

    Parameters:
    spectra (numpy array or list): The dataset containing spectra.
    num_spectra (int): The number of random spectra to select. Default is 100.

    Returns:
    numpy array: An array containing the randomly selected spectra.
    """
    if isinstance(spectra, np.ndarray):
        num_spectra = spectra.shape[0]
    elif isinstance(spectra, list):
        num_spectra = len(spectra)
    else:
        raise TypeError("Input should be a numpy array or a list.")

    random_indices = random.sample(range(num_spectra), num_spectra_to_plot)
    random_spectra = np.array(spectra)[random_indices, :]

    return random_spectra, random_indices


def max_normalize_each_spectrum(spectra):
    """
    Normalize each spectrum to its maximum value.

    Parameters:
    spectra (numpy array): A 2D array where each row is a spectrum.

    Returns:
    numpy array: Max normalized spectra.
    """
    return spectra / np.max(spectra, axis=1, keepdims=True)


def standardization(spectra):
    mean = np.mean(spectra, axis=1, keepdims=True)
    std = np.std(spectra, axis=1, keepdims=True)
    normalized_data = (spectra - mean) / std
    return normalized_data


def min_max_normalize_entire_dataset(spectra):
    """
    Normalizes the spectra data based on the overall minimum and maximum values.
    Handles cases where entire spectra rows are zero.

    Parameters:
        spectra (np.ndarray or list): A 2D numpy array or list where each row represents a spectrum.

    Returns:
        np.ndarray: A normalized numpy array where all values are scaled between 0 and 1.
    """
    # Ensure input is a numpy array
    spectra = np.array(spectra)

    # Calculate min and max values, excluding completely zero rows
    non_zero_spectra = spectra[~np.all(spectra == 0, axis=1)]
    if non_zero_spectra.size == 0:
        # If all rows are zero, return the original array
        return spectra

    min_val = np.min(non_zero_spectra)
    max_val = np.max(non_zero_spectra)

    if min_val == max_val:
        # Avoid division by zero when all non-zero values are the same
        return np.zeros_like(spectra)

    # Normalize the spectra
    scaled_spectra = (spectra - min_val) / (max_val - min_val)

    return scaled_spectra



def normalize_spectra_by_mean_range(spectra, raman_shift, target_range):
    """
    Normalizes each spectrum by dividing it by the mean value in a specified Raman shift range.

    Parameters:
        spectra (np.ndarray or list): A 2D numpy array or list where each row represents a spectrum.
        raman_shift (np.ndarray or list): A 1D array or list representing the Raman shift axis.
        target_range (tuple): A tuple (start, end) specifying the Raman shift range for mean calculation.

    Returns:
        np.ndarray: A normalized numpy array where each row is scaled by its mean value in the target range.
    """
    # Ensure inputs are numpy arrays
    spectra = np.array(spectra)
    raman_shift = np.array(raman_shift)

    # Identify the indices of the Raman shifts within the target range
    start, end = target_range
    indices = (raman_shift >= start) & (raman_shift <= end)

    if not np.any(indices):
        raise ValueError("The specified target range does not overlap with the Raman shift axis.")

    # Normalize each spectrum by the mean value in the target range
    normalized_spectra = np.copy(spectra)
    for i, spectrum in enumerate(spectra):
        mean_value = np.mean(spectrum[indices])
        if mean_value != 0:  # Avoid division by zero
            normalized_spectra[i] = spectrum / mean_value

    return normalized_spectra


def robust_normalize_entire_dataset(spectra):
    """
    Normalizes the spectra data based on robust statistics to reduce sensitivity to outliers.
    Handles cases where entire spectra rows are zero.

    Parameters:
        spectra (np.ndarray or list): A 2D numpy array or list where each row represents a spectrum.

    Returns:
        np.ndarray: A normalized numpy array where all values are scaled between 0 and 1.
    """
    # Ensure input is a numpy array
    spectra = np.array(spectra)

    # Exclude completely zero rows
    non_zero_spectra = spectra[~np.all(spectra == 0, axis=1)]
    if non_zero_spectra.size == 0:
        # If all rows are zero, return the original array
        return spectra

    # Calculate robust min and max using percentiles
    lower_bound = np.percentile(non_zero_spectra, 0.01)  # 1st percentile
    upper_bound = np.percentile(non_zero_spectra, 99.99)  # 99th percentile

    if lower_bound == upper_bound:
        # Avoid division by zero when all non-zero values fall in the same range
        return np.zeros_like(spectra)

    # Clip values to the lower and upper bounds
    clipped_spectra = np.clip(spectra, lower_bound, upper_bound)

    # Normalize the spectra based on the robust range
    scaled_spectra = (clipped_spectra - lower_bound) / (upper_bound - lower_bound)

    return scaled_spectra


def quasi_rayleigh_normalize_spectra(spectra, rayleigh_values, raman_values):
    # Normalize spectra by Rayleigh peak
    rayleigh_normalized_spectra = [spectra[i] / rayleigh_values[i] for i in range(len(spectra))]
    rayleigh_normalized_raman_value = np.divide(raman_values, rayleigh_values,
                                                out=np.zeros_like(raman_values),
                                                where=rayleigh_values != 0)
    rayleigh_normalized_rayleigh_value = np.divide(rayleigh_values, rayleigh_values,
                                                   out=np.zeros_like(raman_values),
                                                   where=rayleigh_values != 0)

    return rayleigh_normalized_spectra, rayleigh_normalized_raman_value, rayleigh_normalized_rayleigh_value


def smooth_spectra(spectra, window_length=11, poly_order=3):
    """
    Apply Savitzky-Golay filter to one or more spectra.

    Parameters:
    spectra (numpy array): A 1D or 2D array where each row is a spectrum.
    window_length (int): The length of the filter window. Default is 11.
    poly_order (int): The order of the polynomial used to fit the samples. Default is 3.

    Returns:
    numpy array: Smoothed spectrum or spectra.
    """
    if spectra.ndim == 1:
        # If input is a single spectrum (1D array)
        return savgol_filter(spectra, window_length, poly_order)
    elif spectra.ndim == 2:
        # If input is multiple spectra (2D array)
        smoothed_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            smoothed_spectra[i] = savgol_filter(spectra[i], window_length, poly_order)
        return smoothed_spectra
    else:
        raise ValueError("Input spectra should be a 1D or 2D numpy array.")


def trim_spectra_by_raman_shift_range(spectra, raman_shift, lower_limit=None, upper_limit=None):
    """
    Trims the raman_shift and corresponding spectra to include only raman_shift values within the specified range.

    Parameters:
    raman_shift (np.ndarray or list): Array or list of raman_shift values.
    spectra (np.ndarray or list): 2D array or list of spectra corresponding to the raman_shift values.
    lower_limit (float, optional): The lower limit for trimming raman_shift. If None, no lower limit is applied.
    upper_limit (float, optional): The upper limit for trimming raman_shift. If None, no upper limit is applied.

    Returns:
    tuple: Trimmed raman_shift and corresponding trimmed spectra.
    """
    # Convert lists to numpy arrays if necessary
    raman_shift = np.array(raman_shift)
    spectra = np.array(spectra)

    # Determine which indices to keep based on provided limits
    if lower_limit is not None and upper_limit is not None:
        indices_within_limits = np.where((raman_shift >= lower_limit) & (raman_shift < upper_limit))[0]
    elif lower_limit is not None:
        indices_within_limits = np.where(raman_shift >= lower_limit)[0]
    elif upper_limit is not None:
        indices_within_limits = np.where(raman_shift < upper_limit)[0]
    else:
        indices_within_limits = np.arange(len(raman_shift))  # If no limits, use all indices

    trimmed_raman_shift = raman_shift[indices_within_limits]
    trimmed_spectra = spectra[:, indices_within_limits]

    return trimmed_spectra, trimmed_raman_shift


def lieberfit(spectra, total_iterations=100, order_polyfit=10):
    # Convert spectra to a numpy array
    spectra = np.array(spectra)

    number_of_spectra, spectra_length = spectra.shape
    fluorescence_corrected_spectra = np.zeros((number_of_spectra, spectra_length))
    fluorescence_background_collected = np.zeros((number_of_spectra, spectra_length))

    for j in tqdm(range(number_of_spectra), desc="Processing Spectra"):
        fluorescence_background = spectra[j, :].copy()
        for i in range(total_iterations):
            # Perform polynomial fitting
            polyfit_coefficient = np.polyfit(np.arange(1, spectra_length + 1), fluorescence_background, order_polyfit)
            polysfit_spectra = np.polyval(polyfit_coefficient, np.arange(1, spectra_length + 1))
            fluorescence_background = np.minimum(polysfit_spectra, fluorescence_background)

        # Store corrected spectra and background
        fluorescence_corrected_spectra[j, :] = spectra[j, :] - fluorescence_background
        fluorescence_background_collected[j, :] = fluorescence_background

    return fluorescence_corrected_spectra, fluorescence_background_collected


def baseline_correction(spectra, raman_shift, poly_order=3, num_std=0.7):
    baseline_fitter = Baseline(x_data=raman_shift)
    baseline_corrected_spectra = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        baseline, _ = baseline_fitter.imodpoly(spectra[i, :], poly_order=poly_order, num_std=num_std)
        baseline_corrected_spectra[i, :] = spectra[i, :] - baseline
    return baseline_corrected_spectra


def find_intensity_at_target_raman_shift(spectra, raman_shift, target_raman_shift):
    # Check if the target Raman shift is within the range of the raman_shift axis
    if target_raman_shift < min(raman_shift) or target_raman_shift > max(raman_shift):
        raise ValueError("Target Raman shift is outside the range of the provided raman_shift axis.")

    # Find the index of the closest raman shift
    closest_index = np.abs(np.array(raman_shift) - target_raman_shift).argmin()

    # Extract the values from the spectra at the closest Raman shift
    if isinstance(spectra, np.ndarray):
        # If spectra is a numpy array
        if spectra.ndim == 1:  # Single spectrum case
            return spectra[closest_index]
        elif spectra.ndim == 2:  # Multiple spectra case
            return spectra[:, closest_index]
        else:
            raise ValueError("Spectra array has unexpected dimensions.")
    elif isinstance(spectra, list):
        # If spectra is a list of lists
        if all(isinstance(row, list) for row in spectra):
            return [row[closest_index] for row in spectra]
        elif isinstance(spectra[0], (int, float)):  # Single spectrum case
            return spectra[closest_index]
        else:
            raise ValueError("Spectra list has unexpected structure.")
    else:
        raise TypeError("Spectra must be a numpy array or a list of lists.")


def mean_intensity_in_range(spectra, raman_shifts, lower_bound, upper_bound):
    """
    Calculate the mean intensity of spectra over a specified range of Raman shifts.

    Parameters:
    spectra (list of lists or np.array): List or array of spectra (each spectrum is a list or 1D array).
    raman_shifts (list or np.array): Corresponding Raman shift axis.
    lower_bound (float): Lower bound of the Raman shift range.
    upper_bound (float): Upper bound of the Raman shift range.

    Returns:
    np.array: Mean intensity for each spectrum within the specified range.
    """
    # Ensure inputs are numpy arrays
    spectra = np.array(spectra)
    raman_shifts = np.array(raman_shifts)

    # Find indices corresponding to the Raman shift range
    mask = (raman_shifts >= lower_bound) & (raman_shifts <= upper_bound)

    # Calculate the mean intensity for each spectrum within the range
    mean_intensities = np.mean(spectra[:, mask], axis=1)
    std_deviations = np.std(spectra[:, mask], axis=1)

    return mean_intensities, std_deviations


def max_intensity_in_range(spectra, raman_shifts, lower_bound, upper_bound):
    """
    Calculate the mean intensity of spectra over a specified range of Raman shifts.

    Parameters:
    spectra (list of lists or np.array): List or array of spectra (each spectrum is a list or 1D array).
    raman_shifts (list or np.array): Corresponding Raman shift axis.
    lower_bound (float): Lower bound of the Raman shift range.
    upper_bound (float): Upper bound of the Raman shift range.

    Returns:
    np.array: Mean intensity for each spectrum within the specified range.
    """
    # Ensure inputs are numpy arrays
    spectra = np.array(spectra)
    raman_shifts = np.array(raman_shifts)

    # Find indices corresponding to the Raman shift range
    mask = (raman_shifts >= lower_bound) & (raman_shifts <= upper_bound)

    # Calculate the mean intensity for each spectrum within the range
    max_intensity = np.max(spectra[:, mask], axis=1)

    return max_intensity


def find_peaks_in_spectra(spectra, prominence=0.5):
    """
    Finds peaks in each row of each spectra based on prominences and returns their
    positions and prominence values.

    Args:
      spectra: A numpy array, where each row represents a spectrum and each entry in a row is a peak intensity.
      prominence: Threshold for prominence of the peaks.

    Returns:
      A list of tuples, where each tuple contains an array of peak indices and an array of their corresponding prominences.
    """
    peaks_in_spectra = []
    prominence_of_peaks_in_spectra = []
    for spectrum in spectra:
        peaks, properties = find_peaks(spectrum, prominence=prominence)
        prominences = properties['prominences']
        peaks_in_spectra.append(peaks)
        prominence_of_peaks_in_spectra.append(prominences)
    return peaks_in_spectra, prominence_of_peaks_in_spectra


def find_peaks_closest_to_target_raman_shift(spectra, raman_shift, target_raman_shift, prominence_threshold,
                                             tolerance=50):
    """
    Finds the closest peaks, their values, and their raman_shift in the spectra set.

    Args:
      raman_shift: A numpy array containing the raman_shift.
      spectra: List of spectra where each spectrum is a row in a numpy array.
      target_raman_shift: The target raman_shift.
      tolerance: The maximum allowed difference between the target raman_shift and the peak raman_shift.
      prominence_threshold: minimum prominence of the peaks.

    Returns:
      closest_peak_indices: List of indices of the closest peaks for each spectrum.
      closest_peak_values: List of values of the closest peaks for each spectrum.
      closest_peak_raman_shift: List of raman_shift of the closest peaks for each spectrum.
      closest_peak_prominences: List of prominences of the closest peaks for each spectrum.
    """

    peak_indices, prominences = find_peaks_in_spectra(spectra, prominence=prominence_threshold)
    closest_peak_indices, closest_peak_intensities, closest_peak_raman_shifts, closest_peak_prominences = [], [], [], []

    for peak_indices, prominences, spectrum in zip(peak_indices, prominences, spectra):
        closest_peak_index, closest_peak_prominence = 0, 0
        min_difference = float('inf')
        for peak_index, prominence in zip(peak_indices, prominences):
            difference = abs(raman_shift[peak_index] - target_raman_shift)
            if difference < min_difference and difference <= tolerance:
                min_difference = difference
                closest_peak_index = peak_index
                closest_peak_prominence = prominence

        closest_peak_indices.append(closest_peak_index)
        closest_peak_intensities.append(spectrum[closest_peak_index])
        closest_peak_raman_shifts.append(raman_shift[closest_peak_index])
        closest_peak_prominences.append(closest_peak_prominence)

    return closest_peak_indices, closest_peak_intensities, closest_peak_raman_shifts, closest_peak_prominences


def filter_from_index_mapping(input_variable, index_mapping):
    """
    Filters the input list based on the index mapping.
    :param input_variable:
    :param input: can be spectra or peak intensity or peak raman shift etc.
    :param index_mapping: the indexes that are to be used
    :return: filtered input and the non-zero indices
    """
    non_zero_indices = extract_non_zero_values(index_mapping)
    filtered_input = [input_variable[i] for i in non_zero_indices]
    return filtered_input, non_zero_indices


def filter_spectra_from_binary_mapping(input_variable, binary_matrix):
    flattened_matrix = binary_matrix.flatten()
    indices_of_ones = np.where(flattened_matrix == 1)[0]
    filtered_spectra = input_variable[indices_of_ones]

    return filtered_spectra

def remove_outliers_from_spectra(spectra, spectra_indices, square_size, threshold=2):
    # Convert list of spectra to a 2D numpy array for easier manipulation
    spectra_array = np.array(spectra)

    # Calculate the mean and standard deviation along each feature (column-wise)
    means = np.mean(spectra_array, axis=0)
    std_devs = np.std(spectra_array, axis=0)

    # Calculate Z-scores for each spectrum
    z_scores = np.abs((spectra_array - means) / std_devs)

    # Determine outliers: any spectrum with any feature Z-score above the threshold is an outlier
    outlier_flags = np.any(z_scores > threshold, axis=1)

    # Initialize lists to store filtered spectra and indices
    filtered_spectra = []
    filtered_indices = []

    # Loop through each spectrum and corresponding index
    for spectrum, index, is_outlier in zip(spectra, spectra_indices, outlier_flags):
        if not is_outlier:
            # Add to filtered lists if not an outlier
            filtered_spectra.append(spectrum)

        else:
            filtered_indices.append(index)

    # Convert each index to (row, column) for a 70x70 grid
 #   filtered_indices = [(int(index) // int(index_mapping.shape[0]), int(index) % int(index_mapping.shape[0])) for index in filtered_indices]
    filtered_indices = [(int(idx) // square_size, int(idx) % square_size) for idx in filtered_indices]

    return filtered_spectra, filtered_indices


"""
heatmap processing functions
"""


def reshape_to_square_matrix(values):
    """
    Reshapes a list or array of values into a square matrix.

    Parameters:
    values (list or numpy array): Input list or array of values to be reshaped.

    Returns:
    tuple:
        numpy array: Square matrix reshaped from the input values.
        numpy array: Square matrix where each entry is the index from the unreshaped values.
    """
    # Calculate the side length of the square matrix
    square_matrix_side_length = int(np.ceil(np.sqrt(len(values))))

    # Pad the values with zeros if necessary to fit the square shape
    padded_values = np.pad(values, (0, square_matrix_side_length ** 2 - len(values)), mode='constant')

    # Reshape the padded values into a square matrix
    square_matrix = padded_values.reshape(square_matrix_side_length, square_matrix_side_length)

    # Create a matrix of indices
    index_matrix = np.arange(len(values))
    padded_index_matrix = np.pad(index_matrix, (0, square_matrix_side_length ** 2 - len(index_matrix)), mode='constant')
    index_matrix = padded_index_matrix.reshape(square_matrix_side_length, square_matrix_side_length)

    return square_matrix, index_matrix


def reshape_to_square_matrix_with_filtered_indices(values, index_mapping, square_size, padding_value=0):
    filled_array = np.full((square_size, square_size), padding_value, dtype=float)  # Ensure the array can hold floats
    value_index = 0
    for i in range(square_size):
        for j in range(square_size):
            index = index_mapping[i, j]

            if index != 0:  # Skip rejected indices
                if value_index < len(values):
                    filled_array[i, j] = values[value_index]
                    value_index += 1
                else:
                    filled_array[i, j] = padding_value  # Pad with the specified value if out of values

    return filled_array


def index_mapping_update(index_mapping, indices):
    """
    Updates the index mapping with the given indices.
    :param index_mapping: overall index mapping
    :param indices: indices to be updated
    :return: updated index mapping
    """
    for index in indices:
        index_mapping[tuple(index)] = 0

    index_mapping_binary = np.where(index_mapping == 0, 0, 1)

    return index_mapping, index_mapping_binary


def find_indexes_where_matrix_value_is_not_at_target_value(matrix, target, tolerance=20):
    """
    Finds the indexes where the matrix value is zero.

    Parameters:
    matrix (numpy.ndarray): Input matrix to be checked.
    target: The target value.
    tolerance: The maximum allowed difference between the target and the matrix value.

    Returns:
    list: List of indexes where the matrix value is zero.
    """
    filter_indexes = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != target and abs(matrix[i][j] - target) > tolerance:
                filter_indexes.append((i, j))
    return filter_indexes


def extract_non_zero_values(matrix):
    non_zero_values = []
    for row in matrix:
        for value in row:
            if value != 0:
                non_zero_values.append(value)
    return non_zero_values


def impose_index_mapping(matrix, index_mapping):
    """
    Parameters:
    - matrix (np.array): The input matrix with peak values.
    - index_mapping (np.array): The matrix that indicates which indices to keep.

    Returns:
    - np.array: A matrix with index_mapping imposed.
    """
    return np.array([[matrix[i, j] if index_mapping[i, j] != 0 else 0
                      for j in range(matrix.shape[1])]
                     for i in range(matrix.shape[0])])


def remove_small_blobs(matrix, blob_size_threshold=10):
    labeled_matrix, num_features = label(matrix)  # Label the connected components
    objects = find_objects(labeled_matrix)  # Find the objects (slices) in the labeled matrix
    cleaned_matrix = matrix.copy()  # Create a copy of the original matrix to modify
    removed_points = []  # List to keep track of removed points

    # Iterate through each object and check its size
    for i, obj in enumerate(objects):
        # Calculate the size of the object
        blob_size = np.sum(labeled_matrix[obj] == (i + 1))

        # Remove the blob if it's smaller than the threshold
        if blob_size < blob_size_threshold:
            # Find the points to remove
            to_remove = np.argwhere(labeled_matrix == (i + 1))
            removed_points.extend(map(tuple, to_remove))

            # Set the points to zero in the cleaned matrix
            cleaned_matrix[labeled_matrix == (i + 1)] = 0

    return cleaned_matrix, removed_points


def circular_crop_square_matrix(matrix, percentage=50):
    n = matrix.shape[0]
    radius = (percentage / 100.0) * (n / 2)
    data = matrix.copy()
    center = (n // 2, n // 2)
    indexes_to_zero = []

    for i in range(n):
        for j in range(n):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance > radius:
                indexes_to_zero.append((i, j))
                data[i, j] = 0

    return data, indexes_to_zero


def calculate_enhancement_factor(sers_intensity, raman_intensity, concentration_sers, concentration_raman):
    """
    Calculate the enhancement factor of a SERS substrate compared to a Raman control.
    The function can accept single values, lists, or numpy arrays for intensities.
    It returns either a single enhancement factor (if single values are provided) or
    the mean and maximum enhancement factors (if multiple values are provided).

    :param sers_intensity: SERS intensity, can be a single value, list, or numpy array
    :param raman_intensity: Raman intensity, can be a single value, list, or numpy array
    :param concentration_sers: Concentration of the sample in the SERS measurement
    :param concentration_raman: Concentration of the sample in the Raman measurement
    :return: Either a single enhancement factor or a tuple of mean and maximum enhancement factors
    """

    # Convert to numpy arrays if input is not already an array
    if isinstance(sers_intensity, (list, int, float)):
        sers_intensity = np.array(sers_intensity)
    if isinstance(raman_intensity, (list, int, float)):
        raman_intensity = np.array(raman_intensity)

    # Calculate the enhancement factors
    if sers_intensity.size == 1 and raman_intensity.size == 1:
        enhancement_factor = (sers_intensity / raman_intensity) * (concentration_raman / concentration_sers)
        print("The enhancement factor is: {:.2e}".format(enhancement_factor).replace('e', ' x 10^'))
        return enhancement_factor
    else:
        mean_sers_intensity = np.mean(sers_intensity)
        max_sers_intensity = np.max(sers_intensity)
        mean_raman_intensity = np.mean(raman_intensity)

        mean_enhancement_factor = (mean_sers_intensity / mean_raman_intensity) * (concentration_raman / concentration_sers)
        max_enhancement_factor = (max_sers_intensity / mean_raman_intensity) * (concentration_raman / concentration_sers)

        print("The mean enhancement factor is: {:.2e}".format(mean_enhancement_factor))
        print("The maximum enhancement factor is: {:.2e}".format(max_enhancement_factor))

        return mean_enhancement_factor, max_enhancement_factor
