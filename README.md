# pySERS: A Surface-Enhanced Raman Spectroscopy (SERS) Analysis toolkit 
Author: Saransh Arora

## Overview
The library has been developed so far in service of a Low-Cost SERS project focuses on democratizing Surface-Enhanced Raman Spectroscopy (SERS) for widespread use by developing affordable and scalable biosensing solutions. The project encompasses diverse functionalities, including sensitivity analysis, reproducibility assessment, and specific use cases like pesticide and bacteria detection. This repository combines computational tools, experimental data, and analysis pipelines to streamline SERS-based research and applications.

---

## Key Features
- **Affordable Fabrication**: Innovative electrodeposition assembly and substrate preparation techniques.
- **Wide Applicability**: Tools for bacteria detection, pesticide identification, and multiplexing.
- **Data Processing**: Comprehensive utilities for smoothing, normalization, trimming, and baseline correction of Raman spectra.
- **Heatmap and Statistical Analysis**: Visualizations like heatmaps, regression plots, and reproducibility assessments.
- **Custom Configuration**: Flexible configuration files for controlling analysis parameters.

---

## Repository Structure

### 1. **Core Codebase**
- `data_classes.py`: Defines classes like `RamanScan`, `ReproducibilityAnalysis`, and `SensitivityAnalysis`.
- Utility Modules:
  - `raman_data_processing_utils.py`: Functions for spectra interpolation, normalization, and outlier removal.
  - `raman_statistical_utils.py`: Coefficient of variation calculations and statistical summaries.
  - `raman_plotting_utils.py`: Visualization functions including raw spectra, mean spectra, and heatmaps.

### 2. **Scripts**
- Sensitivity Analysis:
  - `sers_sensitivity_analysis_for_each_scan.py`
- Reproducibility Analysis:
  - `sers_reproducibility_analysis_each_scan.py`
  - `reproducibility_for_entire_dataset.py`
- Detection Use Cases:
  - `pesticide_detection_thiram.py`
  - `pesticide_detection_thiabendazole.py`
  - `pesticide_detection_multiplexing.py`
  - `bacteria_detection.py`
  - `methylene_blue_detection.py`
- Characterization:
  - `plasmon_resonance_wavelength_characterization.py`
  - `plasmonic_enhancement_against_deposition_voltage.py`
  .... and so on

### 3. **Correspinding Configuration Files**
- Fine-tune parameters for different experiments.
  - `config_limit_of_detection.py`
  - `config_pesticide_detection_multiplexing.py`
  - `config_bacteria_detection.py`
  - `config_reproducibility.py`

### 4. **Data Directory**
Place your Raman spectra datasets in this folder. Maintain a structured hierarchy to support automated dataset processing.

---

## Getting Started

### Prerequisites
1. Python 3.8 or higher.
2. Required packages (install using `pip install -r requirements.txt`).

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/saransharora96/pySERS.git
pip install -r requirements.txt
```
