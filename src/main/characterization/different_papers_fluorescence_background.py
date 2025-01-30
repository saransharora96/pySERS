import os
import numpy as np
import matplotlib.pyplot as plt
import math
from src.utils import raman_data_processing_utils as raman_data, raman_plotting_utils as raman_plots


"""
Read data from the different papers and plot raw data
"""

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\different_papers_tested"
)
all_spectra, all_raman_shifts = [],[]
for file in os.listdir(dataset_location):
    if file.endswith('.txt'):
        raman_shift, spectra = raman_data.read_horiba_raman_txt_file(os.path.join(dataset_location, file))
        all_spectra.append(spectra)
        all_raman_shifts.append(raman_shift)

n_cols = 2  # Number of columns you want
n_rows = math.ceil(len(all_spectra) / n_cols)  # Calculate rows needed
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axs = axs.flatten()
if len(all_spectra) == 1:  # If only one spectrum, ensure axs is a list
    axs = [axs]  # Ensure axs is a list for consistency in the loop
for i, (spectra, raman_shifts) in enumerate(zip(all_spectra, all_raman_shifts)):
    raman_plots.plot_raw_spectra(spectra, raman_shifts, edge_alpha=0.2, ax=axs[i])
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

"""
Trim raman shift axis to appropriate range
"""
all_trimmed_raman_shifts, all_trimmed_spectra = [], []
for raman_shifts, spectra in zip(all_raman_shifts, all_spectra):
    trimmed_spectra, trimmed_raman_shifts = raman_data.trim_spectra_by_raman_shift_range(spectra, raman_shifts, 0, 1500)
    all_trimmed_raman_shifts.append(trimmed_raman_shifts)
    all_trimmed_spectra.append(trimmed_spectra)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axs = axs.flatten()
if len(all_spectra) == 1:  # If only one spectrum, ensure axs is a list
    axs = [axs]  # Ensure axs is a list for consistency in the loop
for i, (trimmed_spectra, trimmed_raman_shifts) in enumerate(zip(all_trimmed_spectra, all_trimmed_raman_shifts)):
    raman_plots.plot_raw_spectra(trimmed_spectra, trimmed_raman_shifts, edge_alpha=0.2, ax=axs[i])
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
for i in range(len(all_trimmed_spectra)):
    raman_plots.plot_mean_spectra(all_trimmed_spectra[i], all_trimmed_raman_shifts[i])
plt.show()

plt.figure(figsize=(8, 6))
for i in range(len(all_trimmed_spectra)):
    ax = plt.gca()
    spectra_array = raman_data.smooth_spectra(np.array(all_trimmed_spectra[i]),101)
    mean_spectrum = np.mean(spectra_array, axis=0)
    std_spectrum = np.std(spectra_array, axis=0)
    ax.plot(all_trimmed_raman_shifts[i], mean_spectrum, label='Mean Spectrum')
    ax.fill_between(all_trimmed_raman_shifts[i], mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.5,
                    label='Standard Deviation')
    ax.set_title('Mean Spectrum with Standard Deviation')
    ax.set_xlabel('Raman Shift (cm^-1)')
    ax.set_ylabel('Intensity')

plt.savefig(f"{dataset_location}/different_papers_fluorescence_background.pdf", format='pdf')
plt.show()
