import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.stats import kruskal, spearmanr, pearsonr
import csv
import math
from src.utils import raman_data_processing_utils as raman_data, raman_plotting_utils as raman_plots


def read_voltage_against_raman_data(folder_path):
    """
    Reads all .txt files in the specified folder and returns them as numpy arrays.

    Args:
      folder_path: The path to the folder containing the .txt files.

    Returns:
      A list of numpy arrays, where each array corresponds to a .txt file.
    """

    # Get a list of all .txt files in the folder
    txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Initialize a dictionary to store concatenated data arrays for each voltage
    raman_data = {}
    raman_shift_data = {}

    # Loop through each file
    for file in txt_files:
        # Parse the voltage from the file name
        voltage = int(file.split('.')[0][-1:])

        # Load data from the file as a numpy array
        raman_shift = np.loadtxt(file)[0, 2:]  # any one of the files will do!
        spectra = np.loadtxt(file)[1:, 2:]

        # Check if the voltage key already exists in the dictionary
        if voltage not in raman_data:
            # If not, create an empty array for the voltage key
            raman_data[voltage] = np.empty((0, spectra.shape[1]))
            raman_shift_data[voltage] = np.empty((0, raman_shift.shape[0]))

        # Concatenate the data arrays to the existing array corresponding to the voltage key
        raman_data[voltage] = np.concatenate((raman_data[voltage], spectra), axis=0)
        raman_shift_data[voltage] = raman_shift

        # 0 spectra come from circular data collection?
        zero_indices = np.where(np.sum(raman_data[voltage], axis=1) == 0)[0]
        raman_data[voltage] = np.delete(raman_data[voltage], zero_indices, axis=0)

    return raman_shift_data, raman_data


dataset_location = (
    r"D:\OOneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\deposition_voltage_against_enhancement"
)

all_raman_shifts, all_spectra = read_voltage_against_raman_data(dataset_location)

# Print the concatenated data arrays grouped by voltage
for voltage, data in sorted(all_spectra.items()):
    print(f"\nVoltage: # {voltage}")
    print("# of Spectra:", len(data))

print('\n')
remap = {1: 1.5, 2: 3, 3: 5, 4: 7, 5: 9}
all_spectra = {remap[key]: value for key, value in all_spectra.items()}  # Remap the keys
all_raman_shifts = {remap[key]: value for key, value in all_raman_shifts.items()}
for voltage, data in sorted(all_spectra.items()):
    print(f"Voltage: {voltage}")

for voltage in all_spectra.keys():
    spectra = all_spectra[voltage]
    index_mapping = list(range(len(spectra)))
    all_spectra[voltage], _ = raman_data.remove_outliers_from_spectra(spectra, index_mapping, threshold=2)

n_cols = 2  # Number of columns you want
n_rows = math.ceil(len(all_spectra) / n_cols)  # Calculate rows needed
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 2))
axs = axs.flatten()
if len(all_spectra) == 1:  # If only one spectrum, ensure axs is a list
    axs = [axs]  # Ensure axs is a list for consistency in the loop
for i, voltage in enumerate(all_spectra.keys()):
    raman_plots.plot_random_spectra(all_spectra[voltage], all_raman_shifts[voltage], num_spectra_to_plot=100, edge_alpha=0.2, ax=axs[i])
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])
plt.tight_layout()
plt.show()

all_rayleigh_range_raman_shifts, all_rayleigh_range_spectra = {}, {}
all_averaged_rayleigh_range_spectra, all_std_dev_rayleigh_range_spectra = {}, {}
start_raman_shift, end_raman_shift = 70, 100

for voltage, data in sorted(all_spectra.items()):
    trimmed_rayleigh_range_spectra, trimmed_rayleigh_range_raman_shifts = \
        raman_data.trim_spectra_by_raman_shift_range(all_spectra[voltage], all_raman_shifts[voltage], start_raman_shift, end_raman_shift)

    averaged_rayleigh_range_spectra, std_dev_rayleigh_range_spectra = \
        np.mean(trimmed_rayleigh_range_spectra, axis=0), np.std(trimmed_rayleigh_range_spectra, axis=0)

    all_rayleigh_range_raman_shifts[voltage] = trimmed_rayleigh_range_raman_shifts
    all_rayleigh_range_spectra[voltage] = trimmed_rayleigh_range_spectra
    all_averaged_rayleigh_range_spectra[voltage] = averaged_rayleigh_range_spectra
    all_std_dev_rayleigh_range_spectra[voltage] = std_dev_rayleigh_range_spectra

voltages = sorted(all_spectra.keys())
for i, voltage in enumerate(voltages):
    plt.plot(all_rayleigh_range_raman_shifts[voltage], all_averaged_rayleigh_range_spectra[voltage], label=f"Voltage: {voltage}")
    plt.fill_between(all_rayleigh_range_raman_shifts[voltage], all_averaged_rayleigh_range_spectra[voltage] - all_std_dev_rayleigh_range_spectra[voltage], all_averaged_rayleigh_range_spectra[voltage] + all_std_dev_rayleigh_range_spectra[voltage],
                     alpha=0.2)
plt.legend()
plt.xlabel('Raman Shift (cm$^{-1}$)')
plt.ylabel('Intensity (a.u.)')
plt.title('Raman Spectra against deposition voltage')
fig.savefig(f"{dataset_location}/deposition_voltage_against_enhancement.pdf", format='pdf')
plt.show()

"""
Export the maxima to a CSV file
"""
filename = "maxima_vs_voltage.csv"
file_path = os.path.join(dataset_location, filename)

maxima = {}
for voltage, data in sorted(all_rayleigh_range_spectra.items()):
    voltage_maxima = [spectrum[np.argmax(spectrum)] for spectrum in data]
    maxima[voltage] = voltage_maxima

csv_rows = [["Voltage", "Maxima"]]
for voltage, maxima_list in maxima.items():
    for maximum in maxima_list:
        csv_rows.append([voltage, maximum])

with open(file_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(csv_rows)

print(f"\nMaxima vs. voltage data exported to CSV file: {file_path}")

for key, value in maxima.items():
    maxima[key] = np.array(maxima[key])


"""
Statistical analysis
"""

# Prepare data for correlation test: flatten arrays and create a corresponding voltage array
all_intensities = np.concatenate(list(maxima.values()))
voltage_levels = np.concatenate([[key] * len(value) for key, value in maxima.items()])

# Calculate Spearman's rank correlation
corr_spearman, p_value_spearman = spearmanr(voltage_levels, all_intensities)
print("\nSpearman's Rank Correlation:")
print(f"Correlation coefficient: {corr_spearman}, P-value: {p_value_spearman}")
if corr_spearman > 0:
    if corr_spearman > 0.5:
        print("Strong positive correlation between voltage and intensity.")
    else:
        print("Weak positive correlation between voltage and intensity.")
elif corr_spearman < 0:
    if corr_spearman < -0.5:
        print("Strong negative correlation between voltage and intensity.")
    else:
        print("Weak negative correlation between voltage and intensity.")
else:
    print("No correlation between voltage and intensity.")

# Calculate Pearson's correlation
corr_pearson, p_value_pearson = pearsonr(voltage_levels, all_intensities)
print("\nPearson's Correlation:")
print(f"Correlation coefficient: {corr_pearson}, P-value: {p_value_pearson}")
if corr_pearson > 0:
    if corr_pearson > 0.5:
        print("Strong positive correlation between voltage and intensity.")
    else:
        print("Weak positive correlation between voltage and intensity.")
elif corr_pearson < 0:
    if corr_pearson < -0.5:
        print("Strong negative correlation between voltage and intensity.")
    else:
        print("Weak negative correlation between voltage and intensity.")
else:
    print("No correlation between voltage and intensity.")
