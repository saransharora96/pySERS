from utils import raman_data_processing_utils as rd, raman_plotting_utils as rp
import os
import numpy as np
import matplotlib.pyplot as plt

dataset_location = (
        r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
        r"\control_substrate_spreading"
    )

INTENSITY_MAP_COLORS = [
    (0, '#FFFFFF'),  # White at position 0
    (0.5, '#008ECC'),  # Light Blue at position 0.5
    (1, '#0F52BA')  # Dark Blue at position 1
]
color_map = rp.custom_color_map(INTENSITY_MAP_COLORS)

target_shift = 1624
PROMINENCE_THRESHOLD = 0.002
PEAK_SHIFT_TOLERANCE = 25

for file in os.listdir(dataset_location):
    if file.endswith('.txt'):
        raman_shift, spectra = rd.read_horiba_raman_txt_file(os.path.join(dataset_location, file))

        scaled_spectra = rd.min_max_normalize_entire_dataset(spectra)
        smoothed_spectra = rd.smooth_spectra(scaled_spectra)

        peak_indices, peak_intensities, _, _ = \
            rd.find_peaks_closest_to_target_raman_shift(smoothed_spectra, raman_shift, target_shift,
                                                                PROMINENCE_THRESHOLD, PEAK_SHIFT_TOLERANCE)

        intensities_square, _ = rd.reshape_to_square_matrix(peak_intensities)
        plt.figure()

        vmin = np.percentile(intensities_square, 1)  # 2nd percentile
        vmax = np.percentile(intensities_square, 99.9)  # 98th percentile
        rp.plot_heatmap(intensities_square, color_map=color_map, vmin=0.02, vmax=vmax)
        plt.savefig(f"{dataset_location}/control_substrate.pdf", format='pdf')
        plt.show()
