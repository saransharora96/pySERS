import numpy as np
import os
from src.utils import raman_data_processing_utils as rd
from src.utils import raman_plotting_utils as rp
import matplotlib.pyplot as plt

""" 
Read the data
"""

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\burning_background_against_time"
)

all_raman_shifts, all_spectra = [], []  # Lists to store all raman shifts and spectra
for file in os.listdir(dataset_location):
    if file.endswith('.txt'):
        raman_shift, spectra = rd.read_horiba_raman_txt_file(os.path.join(dataset_location, file))
        spectra = spectra[1:, :]  # Remove the first row of the spectra
        all_raman_shifts.append(raman_shift)
        all_spectra.append(spectra)


"""
Visualize the raw data
"""
num_plots = len(all_spectra)
cols = int(np.sqrt(num_plots))
rows = int(np.ceil(num_plots / cols))

fig = plt.figure(figsize=(15, 10))
for i in range(num_plots):
    ax = fig.add_subplot(rows, cols, i + 1)
    rp.plot_raw_spectra(all_spectra[i], all_raman_shifts[i], edge_alpha=0.2, ax=ax,
                        title='Spectra Set {}'.format(i + 1))
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity (a.u.)')
plt.tight_layout()
plt.show()


"""
Normalize the spectra by the quasi Rayleigh peak
"""

normalized_spectra_sets = []
for i in range(len(all_spectra)):

    _, rayleigh_intensity, _, _ = \
        rd.find_peaks_closest_to_target_raman_shift(all_spectra[i], all_raman_shifts[i], 90, 0, tolerance=15)

    _, raman_intensity, _, _ = \
        rd.find_peaks_closest_to_target_raman_shift(all_spectra[i], all_raman_shifts[i], 1590, 0, tolerance=15)

    rayleigh_normalized_spectra, rayleigh_normalized_raman_intensity, _ = \
        rd.quasi_rayleigh_normalize_spectra(all_spectra[i], rayleigh_intensity, raman_intensity)

    normalized_spectra_sets.append(np.array(rayleigh_normalized_spectra))

fig = plt.figure(figsize=(15, 10))
for i in range(num_plots):
    ax = fig.add_subplot(rows, cols, i + 1)
    rp.plot_raw_spectra(normalized_spectra_sets[i], all_raman_shifts[i], edge_alpha=0.2, ax=ax,
                        title='Normalized spectra set {}'.format(i + 1))
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity (a.u.)')
plt.tight_layout()
plt.show()

"""
Average corresponding spectra from each set
"""
num_spectra = normalized_spectra_sets[0].shape[0]
averaged_spectra = []
for i in range(num_spectra):
    spectra_to_average = [spectra_set[i] for spectra_set in normalized_spectra_sets]
    averaged_spectrum = np.mean(spectra_to_average, axis=0)
    averaged_spectra.append(averaged_spectrum)

plt.figure()
rp.plot_raw_spectra(averaged_spectra, all_raman_shifts[0], edge_alpha=0.2, title='Averaged Spectra')
plt.show()

# Plot the ratios against the index of the averaged spectrum
plt.figure()
plt.plot(np.arange(0.1, 4, 0.1), rd.smooth_spectra(rayleigh_normalized_raman_intensity[0:39],7))
plt.xlabel('Time (sec)')
plt.yticks([])
plt.savefig(f"{dataset_location}/burning_against_time.pdf", format='pdf')
plt.show()
