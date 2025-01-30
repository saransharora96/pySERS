import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import raman_data_processing_utils as rd, raman_plotting_utils as rp
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


colors = [
            (0, '#FFFFFF'),   # White at position 0
            (0.5, '#008ECC'), # Blue at position 0.5
            (1, '#0F52BA')    # Purple at position 1
        ]
color_map = rp.custom_color_map(colors)
circle_percentage = 50
dataset_location = (
    r"D:\OneDrive_JohnsHopkins\Desktop\JohnsHopkins\Projects\PathogenSERS\journal_low_cost_virus\code\data"
    r"\insufficient_quenching_of_paper_fluorescence"
)

# Read all the files
all_raman_shifts, all_spectra = [], []
files = os.listdir(dataset_location)
files.sort()
for file in files:
    if file.endswith('.txt'):
        raman_shift, spectra = rd.read_horiba_raman_txt_file(os.path.join(dataset_location, file))
        all_raman_shifts.append(raman_shift)
        all_spectra.append(spectra)
all_raman_shifts = np.array(all_raman_shifts)  # convert lists to numpy arrays
all_spectra = np.array(all_spectra)

for i, (spectra, raman_shifts) in enumerate(zip(all_spectra, all_raman_shifts)):

    trimmed_spectra, trimmed_raman_shifts = rd.trim_spectra_by_raman_shift_range(spectra, raman_shifts, 0, 1500)

    intensity_at_max_raman_shift = []
    for spectrum in trimmed_spectra:
        intensity_at_max_raman_shift.append(spectrum[-1])

    intensity_at_max_raman_shift_square, index_mapping = rd.reshape_to_square_matrix(
        intensity_at_max_raman_shift
    )

    fig, axs = plt.subplots(1, 1, figsize=(6,6))
    rp.plot_heatmap(intensity_at_max_raman_shift_square, color_map=color_map, ax=None, vmin=10000, vmax=60000)

    intensity_at_max_raman_shift_circle_crop, zero_indices = \
        rd.circular_crop_square_matrix(intensity_at_max_raman_shift_square, percentage=circle_percentage)
    index_mapping, _ = rd.index_mapping_update(index_mapping, zero_indices)

    filtered_spectra, _ = rd.filter_from_index_mapping(trimmed_spectra, index_mapping)

    # Add scale bar
    scale_bar = AnchoredSizeBar(axs.transData, 10, '0.5 mm', 'lower right', pad=0, color='white',
                                frameon=False, size_vertical=2)
    axs.add_artist(scale_bar)

    plt.savefig(f"{dataset_location}/SupplementaryFig3b_{i}.pdf")
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(6, 3))
    rp.plot_mean_spectra(filtered_spectra, trimmed_raman_shifts, ax=axs)
    axs.set_ylim(500, 60000)
    axs.set_yscale('log')
    plt.savefig(f"{dataset_location}/SupplementaryFig3c_{i}.pdf")
    plt.show()
