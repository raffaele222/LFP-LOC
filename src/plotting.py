import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import logging

from scipy.cluster.hierarchy import dendrogram
from pathlib import Path
from scipy.interpolate import griddata

from .utils import reshape


logger = logging.getLogger("lfploc")


def plot_dbscan(k_distances, min_samples, save_folder, custom_format="png"):
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-NN distance')
    plt.title('k-distance Graph for DBSCAN')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"DBSCAN_k_distance.{custom_format}"), format=custom_format, dpi=600)
    plt.close()


def plot_dendogram(best_k, linked, threshold, save_folder, custom_format="png"):
    plt.figure(figsize=(10, 5))
    plt.title(f"Hierarchical Clustering Dendrogram (k={best_k})")
    dendrogram(
        linked,
        truncate_mode='level',
        p=5,
        color_threshold=threshold + 1e-5,
    )
    plt.axhline(y=threshold + 1e-5, c='black', lw=1.5, linestyle='--', label=f'Cut at distance = {threshold:.2f}')
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"Hierarchical_Clustering_Dendrogram.{custom_format}"), format=custom_format, dpi=600)
    plt.close()


def plot_pca(explained_variance, cumulative_variance,n_components_95, save_folder, feature_importance, selected_features, loadings, custom_format="png"):

    # Plot explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Individual Explained Variance')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='s', label='Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Threshold')
    plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"PCA_Explained_Variance.{custom_format}"), format=custom_format, dpi=600)
    plt.close()
    # # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Original Feature')
    plt.xticks(range(len(feature_importance)), list(selected_features.keys()), rotation=45, ha='right')
    plt.ylabel('Feature Importance Score')
    plt.title('Relevance of Initial Features in PCA')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/feature_importance.{custom_format}", format=custom_format)
    plt.close()

    # Plot PCA loadings heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(loadings, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Loading Value')
    plt.xlabel('Original Feature')
    plt.xticks(range(len(feature_importance)), list(selected_features.keys()), rotation=45, ha='right')
    plt.ylabel('Principal Component Index')
    plt.yticks(range(n_components_95), [f'PC {i+1}' for i in range(n_components_95)])
    plt.title('PCA Component Loadings Heatmap')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/pca_loadings_heatmap.{custom_format}", format=custom_format)
    plt.close()


def plot_cluster_labels(features_reduced, full_labels, labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, cmap='viridis', tip_angle=26, x_left=15, x_right=38, y_top=300, y_bottom=13, custom_format="png"):
    if features_reduced.shape[1] >= 3:
        # Plot the first 3 PCA components colored by cluster
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_reduced[:, 0], features_reduced[:, 1], features_reduced[:, 2], c=labels, cmap=cmap)
        ax.set_title('3 components')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter, label='Cluster Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"3D_Clusters.{custom_format}"), format=custom_format, dpi=600)
        plt.close()

    # Plot the first 2 PCA components colored by cluster
    if features_reduced.shape[1] >= 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels, cmap=cmap)
        plt.title('2 components')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Cluster Label')
        plt.savefig(os.path.join(save_folder, f"2D_Clusters.{custom_format}"), format=custom_format, dpi=600)
        plt.close()
        plt.tight_layout()

    # Show the clusters on the probe layout
    clustered_labels = reshape(full_labels, n_ch_per_shank, n_cols, n_shanks)
    clustered_labels = np.insert(clustered_labels, np.arange(1, n_shanks) * n_cols, np.nan, axis=1)

    fig, ax = plt.subplots(figsize=(3, 10))
    im = ax.imshow(clustered_labels, aspect='equal', cmap=cmap, origin='lower')
    ax.axis('off')

    # only keep 2 significant digits in colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.2f')
    # set fontsize of colorbar ticks to 8
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"Clusters_on_probe.{custom_format}"), format=custom_format, dpi=600)
    plt.close()

    # plot the clusters on the actual electrode positions if available
    if len(electrode_positions[0]) == n_ch:
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            electrode_positions[0], electrode_positions[1],
            c=full_labels, cmap='jet', s=8, marker='s'  # square marker, larger size
        )
        # draw probe outline
        for shank in range(n_shanks):
            shank_positions_x = electrode_positions[0][shank*n_ch_per_shank:(shank+1)*n_ch_per_shank].astype(np.float64)
            shank_positions_y = electrode_positions[1][shank*n_ch_per_shank:(shank+1)*n_ch_per_shank].astype(np.float64)
            plt.plot([np.min(shank_positions_x)-x_left, np.max(shank_positions_x)+x_right, np.max(shank_positions_x)+x_right, np.min(shank_positions_x)-x_left, np.min(shank_positions_x)-x_left],
                    [np.min(shank_positions_y)-y_bottom, np.min(shank_positions_y)-y_bottom, np.max(shank_positions_y)+y_top, np.max(shank_positions_y)+y_top, np.min(shank_positions_y)-y_bottom],
                    color='black', linewidth=0.5)

            # draw tip with specified degree angle
            theta = tip_angle / 2
            tip_x = (np.min(shank_positions_x) + np.max(shank_positions_x)) / 2
            tip_lenght = (np.max(shank_positions_x) + x_right - tip_x) / np.tan(np.radians(theta)) 
            tip_y = 0 - y_bottom - tip_lenght
            side_length = tip_lenght / np.cos(np.radians(theta))
            plt.plot([tip_x, np.min(shank_positions_x)-x_left, np.max(shank_positions_x)+x_right, tip_x],
                    [tip_y, np.min(shank_positions_y)-y_bottom, np.min(shank_positions_y)-y_bottom, tip_y],
                    color='black', linewidth=0.5)
            
        #plt.colorbar(scatter, label='Cluster Label')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"Clusters_on_actual_positions.{custom_format}"), format=custom_format, dpi=600, transparent=True)
        plt.close()

        return scatter.norm
    return None

        
def plot_selected_features(selected_features, n_ch_per_shank, n_cols, n_shanks, save_folder, title='Selected Features', static_scale=False, custom_format="png"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    fig, axs = plt.subplots(1, len(selected_features), figsize=(10, 4))

    # slow workaround to get fix solution
    max_value = 0
    for key, feature in selected_features.items():
        if key == "Cluster Labels" or "Component" in key:
            continue
        reshaped_feature = reshape(feature, n_ch_per_shank, n_cols, n_shanks)
        # add 2 emptu columns between shanks for better visualization
        reshaped_feature = np.insert(reshaped_feature, np.arange(1, n_shanks) * n_cols, np.nan, axis=1)
        max_feature = np.nanmax(reshaped_feature)
        if max_feature > max_value:
            max_value = max_feature
    
    for i, (key, feature) in enumerate(selected_features.items()):
        reshaped_feature = reshape(feature, n_ch_per_shank, n_cols, n_shanks)
        # add 2 emptu columns between shanks for better visualization
        reshaped_feature = np.insert(reshaped_feature, np.arange(1, n_shanks) * n_cols, np.nan, axis=1)
        if key == "Cluster Labels" or "Component" in key:
            im = axs[i].imshow(reshaped_feature, aspect='equal', cmap='viridis', origin='lower')
        else:
            if static_scale:
                im = axs[i].imshow(reshaped_feature, aspect='equal', cmap='viridis', origin='lower', vmin=0, vmax=max_value)
            else:
                im = axs[i].imshow(reshaped_feature, aspect='equal', cmap='viridis', origin='lower')
        axs[i].set_title(key, fontsize=10)
        axs[i].axis('off')  # Turn off axis for cleaner look

        # only keep 3 significant difits in colorbar
        cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04,  format='%.3f')
        # set fontsize of colorbar ticks to 8
        cbar.ax.tick_params(labelsize=8)
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_folder, f'{title}.{custom_format}'), format=custom_format)
    plt.close()

def interpolate_features_to_grid(selected_features, electrode_positions, grid_step_um=15, method='linear'):
    """
    Interpolate 1D per-channel features onto a regular 2D grid defined by electrode positions.
    
    Parameters
    ----------
    selected_features : dict[str, np.ndarray]
        Dict mapping feature name -> 1D array (n_ch,)
    electrode_positions : list or tuple of length 2
        [x_positions, y_positions] each shape (n_ch,)
    grid_step_um : float
        Grid spacing in 'um' units of the electrode_positions.
    method : {'linear', 'cubic', 'nearest'}
        Interpolation method for griddata.
        
    Returns
    -------
    grid_x, grid_y : 2D arrays
        Meshgrid coordinates.
    grid_features : dict[str, 2D array]
        Dict mapping feature name -> interpolated 2D array on grid.
    """
    x = np.asarray(electrode_positions[0])
    if len(electrode_positions[1]) == 768:
        logger.info("Dual probe np detected. Adjusting y_min / y_max")
        y = np.concatenate((np.asarray(electrode_positions[1][:384]), np.asarray(electrode_positions[1][:384])), axis=0)
    else:
        y = np.asarray(electrode_positions[1])

    # Only use finite points for interpolation
    points = np.column_stack((x, y))

    # Define regular grid that covers probe area
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)


    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + grid_step_um, grid_step_um),
        np.arange(y_min, y_max + grid_step_um, grid_step_um)
    )

    grid_features = {}
    for key, feature in selected_features.items():
        values = np.asarray(feature, dtype=float)

        # Mask out NaNs for interpolation
        valid_mask = np.isfinite(values)
        if np.sum(valid_mask) < 3:
            # Not enough points for interpolation, just fill with NaNs
            grid_values = np.full_like(grid_x, np.nan, dtype=float)
        else:
            grid_values = griddata(
                points[valid_mask],
                values[valid_mask],
                (grid_x, grid_y),
                method=method,
                fill_value=np.nan
            )

        grid_features[key] = grid_values

    return grid_x, grid_y, grid_features

def plot_feature_grid_maps(selected_features, electrode_positions, save_folder, method='linear', grid_step_um = 15, title_prefix='Grid', custom_format="png"):
    """
    Save interpolated grid maps of features as images.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    grid_x, grid_y, grid_features = interpolate_features_to_grid(selected_features, electrode_positions, grid_step_um=grid_step_um, method=method)

    n_features = len(grid_features)
    fig, axs = plt.subplots(1, n_features, figsize=(4 * n_features, 5))

    # Ensure axs is iterable
    if n_features == 1:
        axs = [axs]

    for ax, (key, grid_values) in zip(axs, grid_features.items()):
        im = ax.imshow(
            grid_values,
            origin='lower',
            extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
            aspect='equal',
            cmap='viridis'
        )
        ax.set_title(key, fontsize=10)
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.04, location='right')
        cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{title_prefix}_Interpolated_Grid_Maps.{custom_format}'), format=custom_format, dpi=600)
    plt.close()
    return grid_features

def plot_probe_shank_outline(ax, probe_params, probe_x_px, probe_y_px, resolution_um):
    """
    Plots the outline of each probe shank on the given axis using parameters from the CSV DataFrame.
    Args:
        ax: matplotlib axis to plot on
        df: DataFrame loaded from Cluster_Labels_and_Probe_Params.csv
        probe_x_px: array of probe x positions in atlas pixels
        probe_y_px: array of probe y positions in atlas pixels
        resolution_um: atlas resolution in microns per pixel
    """
    x_left = x_right = 30
    y_bottom = probe_params['y_bottom (um)']
    tip_angle = probe_params['tip_angle (degrees)']
    n_shanks = probe_params['n_shanks']
    n_ch_per_shank = probe_params['n_ch_per_shank']

    for shank in range(n_shanks):
        shank_x = probe_x_px[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] 
        shank_y = probe_y_px[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] 
        min_x = np.min(shank_x) - x_left / resolution_um  
        max_x = np.max(shank_x) + x_right / resolution_um 
        min_y = 0 #np.min(shank_y) - y_top / resolution_um  
        # make the probe arrive to the very top
        max_y = np.max(shank_y) + y_bottom / resolution_um 
        ax.plot([min_x, max_x, max_x, min_x, min_x],
                [min_y, min_y, max_y, max_y, min_y],
                color='black', linewidth=1)
        theta = tip_angle / 2
        tip_x = (min_x + max_x) / 2
        tip_length = (max_x - tip_x) / np.tan(np.radians(theta))
        tip_y = max_y + tip_length
        ax.plot([tip_x, min_x, max_x, tip_x],
                [tip_y, max_y, max_y, tip_y],
                color='black', linewidth=1)


def plot_probe_on_atlas(atlas_image, cluster_colors, norm, probe_x_px, probe_y_px, probe_params, AP, ML, DV, save_path, resolution_um, text_pos, stacked=False, save_to_pickle=False, custom_format="png"):
    
    file_name = str(Path(save_path).stem)

    # Plot anatomy and probe clusters in adjusted position
    fig, ax = plt.subplots(figsize=(8,8))
    if len(atlas_image.shape) == 2:
        ax.imshow(atlas_image, cmap='gray')
    else:
        ax.imshow(atlas_image)
    scatter = ax.scatter(probe_x_px, probe_y_px, c=cluster_colors, norm=norm,  s=2, marker='s', alpha= 0.7)
    plot_probe_shank_outline(ax, probe_params, probe_x_px, probe_y_px, resolution_um)
    ax.text(text_pos[1], text_pos[0], f'AP: {AP} mm\nML: {ML} mm\nDV: {DV} mm', color='white', fontsize=10, )
    plt.axis('off')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format=custom_format)

    if save_to_pickle:
        pickle.dump((fig, ax), open(str(Path(save_path).parent) + f"/{file_name}.pickle", "wb"))
    
    plt.axis('off')
    plt.savefig(save_path, dpi=600, bbox_inches='tight', format=custom_format)

    if stacked:
        fig2, ax2 = pickle.load(open(str(Path(save_path).parent) + f"/{file_name.replace('_1', '_0')}.pickle", "rb"))
        scatter = ax2.scatter(probe_x_px, probe_y_px, c=cluster_colors, norm=norm,  s=2, marker='s', alpha= 0.7)
        plot_probe_shank_outline(ax2, probe_params, probe_x_px, probe_y_px, resolution_um)
        ax2.text(text_pos[1], text_pos[0]+50, f'AP: {AP} mm\nML: {ML-1} mm\nDV: {DV} mm', color='white', fontsize=10)
        plt.savefig(str(Path(save_path).parent) + f"/stacked_probes_{file_name}.{custom_format}", dpi=600, bbox_inches='tight', format=custom_format)