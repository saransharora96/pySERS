import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
from utils import raman_data_processing_utils as rd

dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\burning_background_against_intensity"
)

# Lists to store all raman shifts and spectra
all_raman_shifts = []
all_spectra = []

files = os.listdir(dataset_location)
files.sort()
for file in files:
    if file.endswith('.txt'):
        raman_shift, spectra = rd.read_horiba_raman_txt_file(os.path.join(dataset_location, file))
        all_raman_shifts.append(raman_shift)
        all_spectra.append(spectra)

# Convert lists to numpy arrays
all_raman_shifts = np.array(all_raman_shifts)
all_spectra = np.array(all_spectra)

shift = [0, 0.5, 0.5, 0.5, 0.5]

plt.figure(figsize=(8, 6))

# Loop through the arrays and plot each one
for i in range(all_spectra.shape[0]):
    # Extract the Raman shift and spectrum for each sample
    x = all_raman_shifts[i]
    y = all_spectra[i]

    # Trim the X axis values and corresponding Y values
    valid_indices = (x > 1000) & (x < 2000)
    x = x[valid_indices]
    y = y[valid_indices]

    # Apply Savitzky-Golay filter to smooth the spectra
    y = signal.savgol_filter(y, window_length=17, polyorder=3)

    # Normalize and shift spectra for plotting
    plt.plot(x, (y / max(y)) + i * shift[i])

# Add labels and title
plt.xlabel('Raman Shift (cm^-1)')
plt.ylabel('Intensity (a.u.)')
plt.yticks([])
plt.title('Background Burning Spectra')
plt.tight_layout()
plt.savefig(f"{dataset_location}/burning_against_intensity.pdf", format='pdf')
plt.show()
