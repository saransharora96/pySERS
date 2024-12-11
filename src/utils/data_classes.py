import os
import numpy as np
from src.utils import raman_statistical_utils as rs, raman_data_processing_utils as rd
from src.utils.sers_reproducibility_analysis_each_scan import reproducibility_analysis_for_each_map
from src.utils.sers_sensitivity_analysis_for_each_scan import sensitivity_analysis_for_each_map


def read_dataset(directory, analysis_class):
    map_files = []
    print("\nReading files from directory:", directory, "\n")

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                subdir_name = os.path.basename(subdir)
                file_path = os.path.join(subdir, file)
                print("Reading file:", file_path)
                # Instantiate the appropriate analysis class
                map_file = analysis_class(directory, file_path, subdir_name)
                map_file.read_file()
                map_files.append(map_file)

    return map_files


class RamanScan:
    def __init__(self, base_dir, file_path, subdir_name=None):
        self.base_dir = base_dir
        self.subdir_name = subdir_name
        self.file_path = file_path
        self.raman_shift = None
        self.spectra = None
        self.map_size = None

    def read_file(self):
        """Reads the Raman shift data and spectra from a .txt file."""
        try:
            self.raman_shift, self.spectra = rd.read_horiba_raman_txt_file(self.file_path)
            self.map_size = np.int64(np.ceil(np.sqrt(self.spectra.shape[0])))

        except Exception as e:
            print(f"Failed to read file {self.file_path}: {e}")


class ReproducibilityAnalysis(RamanScan):
    def __init__(self, base_dir, file_path, subdir_name=None):
        super().__init__(base_dir, file_path, subdir_name)
        self.filtered_spectra = None
        self.index_mapping = None
        self.intensities_square_dict = None
        self.cov_dict = {}

    def perform_reproducibility_analysis(self, color_scheme=None):
        """Applies the reproducibility analysis script to the read data."""
        if self.raman_shift is not None and self.spectra is not None:
            subdir_path = os.path.join(self.base_dir, self.subdir_name)
            # Unpack the script output
            (self.filtered_spectra,
             self.intensities_square_dict,
             self.index_mapping) = reproducibility_analysis_for_each_map(
                self.spectra,
                self.raman_shift,
                subdir_path,
                color_scheme)
            print(f"\033[93m\nResults for {self.subdir_name}\033[0m")
            for key, value in self.intensities_square_dict.items():
                self.cov_dict[key] = rs.percentage_rsd(value, key)


class SensitivityAnalysis(RamanScan):
    def __init__(self, base_dir, file_path, subdir_name=None):
        super().__init__(base_dir, file_path, subdir_name)
        self.intensity_heatmap = None
        self.digitized_map = None
        self.ratio_map = None
        self.hit_count = None
        self.summed_intensity = None

    def perform_sensitivity_analysis(self,
                                     color_scheme=None,
                                     characteristic_intensity_range=None,
                                     background_intensity_range=None,
                                     noise_intensity_range=None,
                                     std_multiple=None,
                                     show_map_heatmap=False):

        if self.raman_shift is not None and self.spectra is not None:
            subdir_path = os.path.join(self.base_dir, self.subdir_name)

            (self.intensity_heatmap,
             self.ratio_map,
             self.digitized_map,
             self.hit_count,
             self.summed_intensity) = sensitivity_analysis_for_each_map(
                self.spectra,
                self.raman_shift,
                color_scheme,
                characteristic_intensity_range,
                background_intensity_range,
                noise_intensity_range,
                std_multiple,
                show_map_heatmap)
