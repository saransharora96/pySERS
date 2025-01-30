import matplotlib.pyplot as plt
import os
import numpy as np
from utils import raman_plotting_utils as rp

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\plasmon_wavelength_characterization"
)

# Get a list of all CSV files in the folder and read each file into a numpy array
csv_files = [f for f in os.listdir(dataset_location) if f.endswith('.csv')]
data_arrays = []

for csv_file in csv_files:
    file_path = os.path.join(dataset_location, csv_file)
    data_array = np.genfromtxt(file_path, delimiter=',', skip_header=1)
    data_arrays.append(data_array)

# Determine the number of subplots needed
num_subplots = len(data_arrays)

# Create a figure with subplots
fig, axes = plt.subplots(num_subplots, 1, figsize=(6, num_subplots * 2), sharex=True)

# Plot each spectrum as a subplot
for i, data_array in enumerate(data_arrays):
    axes[i].plot(data_array[:, 0], data_array[:, 1])
    axes[i].set_ylabel('Intensity')
    if i == num_subplots - 1:
        axes[i].set_xlabel('Raman Shift (cm^-1)')
    axes[i].set_title(f'Spectrum {i + 1}')

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()
# Trim all the data in data_arrays to a range of 250 to 800 in the X axis
trimmed_data_arrays = []

for data_array in data_arrays:
    trimmed_data = data_array[(data_array[:, 0] >= 250) & (data_array[:, 0] <= 800)]
    trimmed_data_arrays.append(trimmed_data)

# Stack all the trimmed spectra into a single array
spectra = np.array([data[:, 1] for data in trimmed_data_arrays])

# Extract the Raman shift values from the first trimmed data array
wavelengths = trimmed_data_arrays[0][:, 0]

fig, ax = plt.subplots()
rp.plot_mean_spectra(spectra, wavelengths, ax=ax)
fig.savefig(f"{dataset_location}/plasmonic_resonance_wavelength_characterization.pdf", format='pdf')
plt.show()

