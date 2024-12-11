import numpy as np

INSTALLATION_NEEDED = False

# Data processing parameters
TRIM_RAMAN_SHIFT_RANGE = [1000, 1700]
COMMON_SHIFT = np.linspace(1000, 1700, 255)
CHARACTERISTIC_INTENSITY_RANGE = [1610, 1640]
BACKGROUND_INTENSITY_RANGE = [1500, 1750]
NOISE_INTENSITY_RANGE = [1100, 1400]
STD_MULTIPLE = 4

# Flags for results for each map
SHOW_MAP_HEATMAP = False
SHOW_SPECTRA_PLOTS = False

# Flags for results for the entire dataset
SHOW_DATASET_HEATMAPS = True
SHOW_DATASET_SPECTRA_PLOTS = True
SHOW_DATASET_REGRESSION_PLOTS = True
SAVE_DATASET_FIGURES = True

INTENSITY_MAP_COLORS = [
    (0, '#FFFFFF'),   # White at position 0
    (0.1, '#B3E5FC'), # Light Sky Blue at position 0.2
    (0.25, '#4FC3F7'), # Sky Blue at position 0.4
    (0.4, '#008ECC'), # Light Blue at position 0.6
    (0.6, '#003F8C'), # Medium Blue at position 0.8
    (1, '#0F52BA')    # Dark Blue at position 1
]
DIGITAL_MAP_COLORS = ['#FFFFFF', '#008ECC']



