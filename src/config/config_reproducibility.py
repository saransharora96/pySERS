# Flags for results for each map
SHOW_MAP_SPECTRA_PLOTS = False
SHOW_MAP_PEAK_PLOTS = False
SHOW_MAP_HEATMAP = False
SHOW_MAP_RSD_PLOTS = False
REPORT_MAP_PERCENTAGE_RSD = False
SAVE_MAP_FIGURES = False

# Flags for results for the entire dataset
SHOW_DATASET_CONTOUR_PLOTS = False
SHOW_DATASET_SPECTRA_PLOTS = False
SHOW_DATASET_HEATMAPS = False
SAVE_DATASET_FIGURES = False
SAVE_DATASET_CSV = False

INSTALLATION_NEEDED = False

# Hyperparameters for the program
PROMINENCE_THRESHOLD = 0.002
MAP_COLORS = [
    (0, '#FFFFFF'),  # White at position 0
    (0.5, '#008ECC'),  # Light Blue at position 0.5
    (1, '#0F52BA')  # Dark Blue at position 1
]
Z_THRESHOLD = 4
BLOB_SIZE_THRESHOLD = 100
PEAK_SHIFT_TOLERANCE = 25
TARGET_RAMAN_SHIFTS = [85, 450, 770, 1624]
COLOR_BAR_RANGE = [(0.22,0.33), (0.33,0.42), (0.22,0.28), (0.45,0.50), (0.8,0.95)]
