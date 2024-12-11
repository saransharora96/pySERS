import numpy as np

INSTALLATION_NEEDED = False

# Data processing parameters
TRIM_RAMAN_SHIFT_RANGE = [700, 1800]
CHARACTERISTIC_INTENSITY_RANGE = [775, 795]
BACKGROUND_INTENSITY_RANGE = [765,825]
NOISE_INTENSITY_RANGE = [1800, 1900]
STD_MULTIPLE = 4

# Flags for results for each map
SHOW_MAP_HEATMAP = False
SHOW_SPECTRA_PLOTS = False

# Flags for results for the entire dataset
SHOW_DATASET_HEATMAPS = True
SHOW_DATASET_SPECTRA_PLOTS = True
SHOW_DATASET_REGRESSION_PLOTS = True
SAVE_DATASET_FIGURES = True

# Color schemes
INTENSITY_MAP_COLORS = [
    (0.0, '#FFFFFF'),  # Lightest shade of red-orange
    (0.15, '#F69D6E'),  # Light orange
    (0.25, '#F18521'),  # Orange-red
    (0.35, '#CB181D'),  # Red
    (1.0, '#67000D')  # Dark red
]
DIGITAL_MAP_COLORS = ['#FFFFFF', '#F18521']
