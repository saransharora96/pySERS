import os
from src.utils import raman_data_processing_utils as rd
from src.utils import raman_plotting_utils as rp
import matplotlib.pyplot as plt

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\enhancement_factor"
)

files = os.listdir(dataset_location)
data = {}  # Dictionary to store the data

concentration_sers = 10e-6
concentration_raman = 10e-2

"""
Read and pre-process data
"""
for file in files:
    if file.endswith('.txt'):

        print(file)

        file_path = os.path.join(dataset_location, file)
        raman_shift, spectra = rd.read_horiba_raman_txt_file(file_path)

        spectra, raman_shift = rd.trim_spectra_by_raman_shift_range(spectra, raman_shift, 350, None)
        spectra, _ = rd.lieberfit(spectra, total_iterations=100, order_polyfit=10)

        data[file] = raman_shift, spectra

        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        rp.plot_raw_spectra(spectra, raman_shift, edge_alpha=0.2, ax=axs)
        plt.show()

_, intensity_sers, _, _ = rd.find_peaks_closest_to_target_raman_shift(data['substrate.txt'][1], data['substrate.txt'][0],
                                                                      target_raman_shift=1624, prominence_threshold=100,
                                                                      tolerance=15)
_, intensity_raman, _, _ = rd.find_peaks_closest_to_target_raman_shift(data['control.txt'][1], data['control.txt'][0],
                                                                       target_raman_shift=1624, prominence_threshold=100,
                                                                       tolerance=15)
rd.calculate_enhancement_factor(intensity_sers, intensity_raman, concentration_sers, concentration_raman)
