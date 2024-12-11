import numpy as np

INSTALLATION_NEEDED = False
STD_MULTIPLE = 4

# Flags for results for each map
SHOW_MAP_HEATMAP = False
SHOW_SPECTRA_PLOTS = False

# Flags for results for the entire dataset
SHOW_DATASET_HEATMAPS = True
SHOW_DATASET_SPECTRA_PLOTS = True
SHOW_DATASET_REGRESSION_PLOTS = True
SAVE_DATASET_FIGURES = True

# Data processing parameters
CHARACTERISTIC_INTENSITY_RANGE_THIRAM = [1360, 1400]
BACKGROUND_INTENSITY_RANGE_THIRAM = [1300,1450]

CHARACTERISTIC_INTENSITY_RANGE_THIABENDAZOLE = [775, 795]
BACKGROUND_INTENSITY_RANGE_THIABENDAZOLE = [765,825]

NOISE_INTENSITY_RANGE = [1800, 1900]

# Color schemes
INTENSITY_MAP_COLORS_THIRAM = [
    (0.0, '#FFFFFF'),  # Lightest shade of green
    (0.15, '#A1D99B'),  # Light green
    (0.25, '#74C476'),  # Medium green
    (0.35, '#31A354'),  # Deep green
    (1.0, '#00441B')  # Dark green
]
DIGITAL_MAP_COLORS_THIRAM = ['#FFFFFF', '#74C476']

INTENSITY_MAP_COLORS_THIABENDAZOLE = [
    (0.0, '#FFFFFF'),  # Lightest shade of red-orange
    (0.15, '#F69D6E'),  # Light orange
    (0.25, '#F18521'),  # Orange-red
    (0.35, '#CB181D'),  # Red
    (1.0, '#67000D')  # Dark red
]
DIGITAL_MAP_COLORS_THIABENDAZOLE = ['#FFFFFF', '#F18521']
