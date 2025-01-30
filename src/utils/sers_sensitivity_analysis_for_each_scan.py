import numpy as np
import matplotlib.pyplot as plt
from utils import raman_data_processing_utils as rd, raman_plotting_utils as rp


def digital_sers_analysis(spectra, raman_shift, characteristic_intensity_range, background_intensity_range,
                          noise_intensity_range, std_multiple=3):
    characteristic_intensity = rd.max_intensity_in_range(spectra, raman_shift, characteristic_intensity_range[0],
                                                         characteristic_intensity_range[1])
    mean_intensity_background, _ = rd.mean_intensity_in_range(spectra, raman_shift, background_intensity_range[0],
                                                              background_intensity_range[1])
    _, std_intensity_background = rd.mean_intensity_in_range(spectra, raman_shift, noise_intensity_range[0],
                                                             noise_intensity_range[1])
    thresholds = mean_intensity_background + std_multiple * std_intensity_background
    digitized_results = np.where(characteristic_intensity > thresholds, 1, 0)
    digitized_results_square, _ = rd.reshape_to_square_matrix(digitized_results)

    return digitized_results_square, characteristic_intensity-thresholds


def map_intensity_to_background_ratio(spectra, raman_shift, characteristic_intensity_range, background_intensity_range):
    characteristic_intensity = rd.max_intensity_in_range(spectra, raman_shift, characteristic_intensity_range[0],
                                                         characteristic_intensity_range[1])
    mean_intensity_background, std_intensity_background = rd.mean_intensity_in_range(spectra, raman_shift,
                                                                                     background_intensity_range[0],
                                                                                     background_intensity_range[1])
    ratio = characteristic_intensity / mean_intensity_background
    ratio_square, _ = rd.reshape_to_square_matrix(ratio)
    return ratio_square


def count_ones(binary_mapping):
    if isinstance(binary_mapping, np.ndarray):
        return np.sum(binary_mapping == 1)
    else:
        return sum(1 for value in binary_mapping if value == 1)


def sum_where_one(values_matrix, binary_mapping):
    if binary_mapping.shape != values_matrix.shape:
        raise ValueError("The binary mapping and values matrix must have the same shape.")
    return np.sum(values_matrix * binary_mapping)


def sensitivity_analysis_for_each_map(spectra, raman_shift, color_map='viridis',
                                      characteristic_intensity_range=None, background_intensity_range=None,
                                      noise_intensity_range=None, std_multiple=None, show_map_heatmap=False):
    scaled_spectra = rd.robust_normalize_entire_dataset(spectra)
    ratio_square = map_intensity_to_background_ratio(scaled_spectra, raman_shift, characteristic_intensity_range,
                                                     background_intensity_range)

    # Digital SERS analysis
    digitized_results, characteristic_intensity = digital_sers_analysis(spectra, raman_shift, characteristic_intensity_range,
                                              background_intensity_range, noise_intensity_range, std_multiple)
    characteristic_intensity_square, _ = rd.reshape_to_square_matrix(characteristic_intensity)
    hit_count = count_ones(digitized_results)
    summed_intensity = sum_where_one(characteristic_intensity_square, digitized_results)
    print(f"\033[94mThe hit count in the bulk is {hit_count}\033[0m")
    print(f"\033[94mThe sum of intensities @ hits in the bulk is {summed_intensity:.2f} a.u. \033[0m\n")

    if show_map_heatmap is True:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        rp.plot_heatmap(ratio_square, color_map=color_map, ax=axs[0], vmin=1, vmax=4)
        rp.plot_heatmap(digitized_results, color_map=color_map, ax=axs[1], vmin=0, vmax=1)
        plt.show()

    return characteristic_intensity_square, ratio_square, digitized_results, hit_count, summed_intensity
