import numpy as np
import math
import matplotlib.pyplot as plt
import os

from pathlib import Path
from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt, welch


def bandpass_filter(
        data : np.ndarray,
        lowcut : int | float,
        highcut : int | float,
        fs : int,
        order : int = 2
) -> np.ndarray:
    
    # compute and apply bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=1)
    return y


def build_border_image(annotated_image):
    border_mask = np.zeros_like(annotated_image, dtype=bool)
    border_mask[1:, :] |= annotated_image[1:, :] != annotated_image[:-1, :]
    border_mask[:-1, :] |= annotated_image[:-1, :] != annotated_image[1:, :]
    border_mask[:, 1:] |= annotated_image[:, 1:] != annotated_image[:, :-1]
    border_mask[:, :-1] |= annotated_image[:, :-1] != annotated_image[:, 1:]

    border_image = np.full((*annotated_image.shape, 3), 255, dtype=np.uint8)
    border_image[border_mask] = 0
    return border_image


def cluster_color_map_for_labels(labels, cmap_name='Blues', min_shade=0.3):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if unique.size == 0:
        return {}, []
    # Sort unique labels so mapping is deterministic
    unique_sorted = np.sort(unique)
    cmap = plt.cm.get_cmap(cmap_name)
    shades = cmap(np.linspace(min_shade, 1.0, len(unique_sorted)))
    cluster_to_color = {lab: tuple(shades[i]) for i, lab in enumerate(unique_sorted)}
    colors = [cluster_to_color[int(l)] for l in labels]
    return cluster_to_color, colors


def compute_psd(
        data : np.ndarray,
        fs : int
) -> tuple[np.ndarray, np.ndarray]:
    
    # define the time segment on which to perform the psd using the Welch method
    t_segment = 8 # time segment in seconds
    
    # if the recording is not downsampled, downsample it to speed up psd calculation
    if fs != 1250:
        dec_factor = int(fs // 1250)
        data = data[:, ::dec_factor]
        fs = 1250
    
    n_per_seg = int(2 ** round(math.log2(int(t_segment*fs))))

        
    n_per_seg = int(2 ** round(math.log2(int(t_segment*fs))))

    # computes power spectral density
    f, Pxx = welch(data, fs=fs, nperseg=n_per_seg, axis=1)
    # Compute Pxx in dB/Hz
    return f, Pxx


def configure_axes_in_um(ax, image_shape, resolution_um):
    """
    Show axis coordinates in um with 1 mm spacing and overlay a grid.
    """
    height_px, width_px = image_shape[:2]
    tick_step_px = max(1, int(round(1000 / resolution_um)))
    x_ticks = np.arange(0, width_px, tick_step_px, dtype=int)
    y_ticks = np.arange(0, height_px, tick_step_px, dtype=int)
    x_labels_um = (x_ticks * resolution_um).astype(int)
    y_labels_um = (y_ticks * resolution_um).astype(int)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels_um)
    ax.set_yticklabels(y_labels_um)
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")

    ax.grid(True, color="black", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(False)


def get_label_order(cluster_labels, n_shanks, n_ch_per_shank):
    unique_labels = np.unique(cluster_labels)
    centroid_pos = np.zeros((len(unique_labels),n_shanks))
    for shank in range(n_shanks):
        labels_shank = cluster_labels[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] 
        for j,label in enumerate(unique_labels):
            centroid_pos[j, shank] = np.mean(np.where(labels_shank == label)[0])
    return np.nanmean(centroid_pos, axis=1)


def get_probe_properties(
        electrode_positions : np.ndarray,
        num_shanks : int
):  
    x_values = electrode_positions[0]

    sorted_x_values = np.sort(x_values)
    diffs = np.diff(sorted_x_values)

    if num_shanks > 1:
        shank_spacing = np.max(diffs)
    else:
        shank_spacing = np.max(diffs) + 1

    n_cols = 0
    for x_diff in np.unique(diffs):
        if x_diff >= shank_spacing:
            break
        n_cols += 1

    if not num_shanks > 1:
        shank_spacing = 0

    return n_cols, shank_spacing


def get_relabeled_atlas(atlas_id : str, download_if_not_available : bool = False):

    config_folder = Path.home() / ".config" / "lfploc"

    if not os.path.isdir(config_folder):
        if not download_if_not_available:
            return None, None
        
        os.makedirs(config_folder.parent, exist_ok=True)
        os.makedirs(config_folder, exist_ok=True)

    if os.path.exists(config_folder / atlas_id / "relabeled_atlas.npy") and os.path.exists(config_folder / atlas_id / "rgb_map.npy"):
        relabeled_atlas = np.load(config_folder / atlas_id / "relabeled_atlas.npy")
        rgb_map = np.load(config_folder / atlas_id / "rgb_map.npy", allow_pickle=True).item()
        return relabeled_atlas, rgb_map
    
    print("Downloading of relabeled atlases not available yet. As of currently, a relabeled version of the 'kim_mouse_isotropic_20um' can be found in the GitHub repository under the 'lib' directory")
    print("To enable, download the two 'relabaled_atlas.npy' and 'rgb_map.npy' files and copy them to the following directory:")
    print(config_folder / "kim_mouse_isotropic_20um")
    raise NotImplementedError


def interpolate_features_to_grid(selected_features, electrode_positions, grid_step_um=15, method='linear'):
    x = np.asarray(electrode_positions[0])
    if len(electrode_positions[1]) == 768:
        print("Dual probe np detected. Adjusting y_min / y_max")
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


def remap_labels(cluster_labels, probe_params):
    unique_labels = np.unique(cluster_labels)
    n_shanks = probe_params['n_shanks']
    n_ch_per_shank = probe_params['n_ch_per_shank']
    label_order = get_label_order(cluster_labels, n_shanks, n_ch_per_shank) # get order of label from the top down

    label_count = np.array([np.sum(cluster_labels == label) for label in unique_labels])

    # # Combine counts and labels, then sort by order (from top to bottom appearance)
    sorted_pairs = sorted(zip(label_order, label_count, unique_labels), reverse=True)
    sorted_labels = [label for label_order, count, label in sorted_pairs]

    # Re-map cluster labels to new sequential labels based on sorted order
    remapped_labels = np.zeros_like(cluster_labels)
    for new_label, old_label in enumerate(sorted_labels):
        remapped_labels[cluster_labels == old_label] = new_label + 1  # Start from 1
    return remapped_labels, sorted_labels


def reshape(
        array : np.ndarray,
        nCh_per_shank : int,
        n_cols : int,
        n_shanks : int
) -> np.ndarray:
    # split array by shanks, reshape each shank and concatenate
    reshaped_array = np.full((nCh_per_shank//n_cols, n_cols * n_shanks), np.nan)
    for shank in range(n_shanks):
        shank_array = array[shank*nCh_per_shank:(shank+1)*nCh_per_shank]
        reshaped_array[:, shank*n_cols:(shank+1)*n_cols] = reshape_shank(shank_array, (nCh_per_shank//n_cols, n_cols))
    return reshaped_array

    
def reshape_shank(
        array : np.ndarray,
        new_shape : int | tuple
) -> np.ndarray:
    # place first element in (0,0), second in (0,1), ..., 129th in (1,0), 130th in (1,1) and so on
    new_array = np.full(new_shape, np.nan)
    for i in range(len(array)):
        new_array[i // new_shape[1], i % new_shape[1]] = array[i]
    return new_array