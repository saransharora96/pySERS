import numpy as np

INSTALLATION_NEEDED = False

# Data processing parameters
CHARACTERISTIC_INTENSITY_RANGE = [1360, 1400]
BACKGROUND_INTENSITY_RANGE = [1300,1450]
NOISE_INTENSITY_RANGE = [1700, 1800]
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
    (0.0, '#FFFFFF'),  # Lightest shade of green
    (0.15, '#A1D99B'),  # Light green
    (0.25, '#74C476'),  # Medium green
    (0.35, '#31A354'),  # Deep green
    (1.0, '#00441B')  # Dark green
]
DIGITAL_MAP_COLORS = ['#FFFFFF', '#74C476']
