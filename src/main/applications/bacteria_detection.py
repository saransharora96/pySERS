import time
import matplotlib.pyplot as plt
from utils import raman_plotting_utils as rp, raman_data_processing_utils as rd
from install_modules import upgrade_pip, install_packages
from utils.data_classes import SensitivityAnalysis, read_dataset
from config.config_pesticide_detection_thiram import (INSTALLATION_NEEDED)
import numpy as np
import os
import warnings
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable

# Suppress specific warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")


def visualize_results(embedding, labels, kmeans_model, categories=None, marker_sizes=None, marker_shapes=None, category_colors=None):
    """
    Visualize UMAP results with k-Means decision boundaries and cluster centers.

    Parameters:
    - embedding (numpy.ndarray): 2D array (n_samples x 2) representing reduced data.
    - labels (numpy.ndarray): True labels for each sample.
    - kmeans_model (KMeans): Trained k-Means model.
    - categories (list): List of category names corresponding to `labels` (optional).
    - marker_sizes (list or numpy.ndarray): List of marker sizes for each category (optional).
    - marker_shapes (list): List of marker shapes for each category (optional).

    Returns:
    - None: Displays the visualization.
    """
    # Default marker sizes and shapes if not provided
    if marker_sizes is None:
        marker_sizes = [100] * len(np.unique(labels))
    if marker_shapes is None:
        marker_shapes = ['o', 'D', 'x', '^', 'v', 's']
    if category_colors is None:
        category_colors = ['#F18521', '#1A99D6']

    custom_cmap = ListedColormap(category_colors)
    # Create a grid for decision boundary visualization
    x_min, x_max = embedding[:, 0].min() - 1, embedding[:, 0].max() + 1
    y_min, y_max = embedding[:, 1].min() - 1, embedding[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure size

    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.1, cmap=custom_cmap)

    # Plot UMAP embedding with true labels
    unique_labels = np.unique(labels)
    scatter_points = []  # To store scatter plot handles for legend
    for i, label in enumerate(unique_labels):
        idx = labels == label
        scatter = ax.scatter(
            embedding[idx, 0], embedding[idx, 1],
            label=[""],
            s=marker_sizes[i % len(marker_sizes)],  # Custom marker size
            marker=marker_shapes[i % len(marker_shapes)],  # Custom marker shape
            c=category_colors[i % len(category_colors)],  # Use specified hex color
            edgecolor='none', 
            alpha=0.5
        )
        scatter_points.append(scatter)

        # Add legend
    ax.legend(
        scatter_points + [plt.Line2D([], [], color='red', marker='x', linestyle='None', markersize=10)],
        [*categories],
        frameon=False,
    )
    # Adjust plot aesthetics
    ax.set_aspect('equal', adjustable='box')  # Square aspect ratio
    ax.spines['top'].set_visible(False)      # Remove top spine
    ax.spines['right'].set_visible(False)    # Remove right spine
    ax.spines['left'].set_visible(False)     # Remove left spine
    ax.spines['bottom'].set_visible(False)   # Remove bottom spine
    ax.set_xticks([])                        # Remove x-axis ticks
    ax.set_yticks([])                        # Remove y-axis ticks

    plt.savefig(os.path.join(output_path, 'bacteria_umap.svg'), format='svg')
    plt.show()


def combine_spectra(bsubtilis_data):
    combined_spectra = []
    combined_raman_shift = None

    for raman_map in bsubtilis_data:
        if combined_raman_shift is None:
            combined_raman_shift = raman_map.raman_shift
        combined_spectra.extend(raman_map.spectra)

    combined_spectra = np.array(combined_spectra)
    return combined_spectra, combined_raman_shift


if __name__ == "__main__":

    start_time = time.time()  # Record the start time

    if INSTALLATION_NEEDED:
        upgrade_pip()
        install_packages()

    bsubtilis_data_path = (
        "D:/OneDrive_JohnsHopkins/Desktop/JohnsHopkins/Projects/PathogenSERS/journal_low_cost_virus/pySERS/data/bacteria_detection/2025_01_27 high concentration comparison/processed/b_subtilis"
    )
    ecoli_data_path = (
        "D:/OneDrive_JohnsHopkins/Desktop/JohnsHopkins/Projects/PathogenSERS/journal_low_cost_virus/pySERS/data/bacteria_detection/2025_01_27 high concentration comparison/processed/e_coli"
    )
    output_path = (
        "D:/OneDrive_JohnsHopkins/Desktop/JohnsHopkins/Projects/PathogenSERS/journal_low_cost_virus/pySERS/data/bacteria_detection/"
    )

    bsubtilis_data = read_dataset(bsubtilis_data_path, SensitivityAnalysis)
    ecoli_data = read_dataset(ecoli_data_path, SensitivityAnalysis)

    # Data pre-processing and mean spectra plotting
    for raman_map in bsubtilis_data:
        raman_map.spectra, raman_map.raman_shift = rd.trim_spectra_by_raman_shift_range(raman_map.spectra, raman_map.raman_shift, lower_limit=None, upper_limit=1700)

    file_number = 1
    for raman_map in ecoli_data:
        raman_map.spectra, raman_map.raman_shift = rd.trim_spectra_by_raman_shift_range(raman_map.spectra, raman_map.raman_shift, lower_limit=None, upper_limit=1700)
        if file_number != 3: 
            raman_map.spectra = rd.remove_outliers_from_spectra(raman_map.spectra, threshold=2)
        file_number += 1
        
    bsubtilis_spectra, raman_shift = combine_spectra(bsubtilis_data)
    bsubtilis_spectra = rd.smooth_spectra(bsubtilis_spectra, window_length=7, poly_order=3)
    bsubtilis_spectra = rd.min_max_normalize_entire_dataset(bsubtilis_spectra)

    ecoli_spectra, _ = combine_spectra(ecoli_data)
    ecoli_spectra = rd.smooth_spectra(ecoli_spectra, window_length=7, poly_order=3)
    ecoli_spectra = rd.min_max_normalize_entire_dataset(ecoli_spectra)

    all_spectra = np.vstack((ecoli_spectra, bsubtilis_spectra))
    ecoli_labels = np.zeros(ecoli_spectra.shape[0], dtype=int)  # All labels 0 for E. coli
    bsubtilis_labels = np.ones(bsubtilis_spectra.shape[0], dtype=int)  # All labels 1 for B. subtilis
    all_labels = np.concatenate((ecoli_labels, bsubtilis_labels))
    categories = ["E. coli", "B. subtilis"]
    
    umap_embedding = rd.compute_umap(all_spectra, n_neighbors=10, min_dist=0.75)
    kmeans_model = rd.compute_kmeans(umap_embedding, n_clusters=2)
    visualize_results(umap_embedding, all_labels, kmeans_model, categories)

    clf, y_test, y_pred = rd.random_forest_classifier_for_raman_spectra(all_spectra, all_labels, raman_shift)
    fig, ax = plt.subplots(figsize=(6, 6))  # Square aspect
    rp.plot_confusion_matrix(ax, y_test, y_pred, ['E. coli', 'B. subtilis'])   
    plt.savefig(os.path.join(output_path, 'bacteria_confusion_matrix.svg'), format='svg')
    plt.show() 
    
    feature_importances = clf.feature_importances_  # Get feature importances from the Random Forest
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    norm, cmap = rp.add_feature_importance_shading(ax, raman_shift, feature_importances, color_hex='#2FA148', max_alpha=0.3)
    rp.plot_mean_spectra(bsubtilis_spectra + 0.5, raman_shift, ax=ax, title='Mean Spectrum with Standard Deviation', label='B. Subtilis')
    rp.plot_mean_spectra(ecoli_spectra, raman_shift, ax=ax, title='Mean Spectrum with Standard Deviation', label='E. Coli')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Dummy array for the ScalarMappable
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Feature Importance")
    ax.legend(frameon=False, loc='upper right')
    ax.set_yticks([])      
    plt.savefig(os.path.join(output_path, 'bacteria_spectra.svg'), format='svg')
    plt.show()

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    
