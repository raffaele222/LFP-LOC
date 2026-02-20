import numpy as np
import h5py
import logging
import math
import os
import matplotlib.pyplot as plt

from scipy.signal import welch, butter, filtfilt
from pathlib import Path


logger = logging.getLogger("lfploc")


def already_filtered(
        file_path : str
) -> bool:
    
    if not "h5" in file_path:
        return False
    
    f = h5py.File(file_path)
    if not "SpikeLAB" in f.keys():
        return False
    if f.require_group("SpikeLAB").get("Bandwidth")[...][1] >= 500:
        return False
    return True


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


def cluster_color_map_for_labels(labels, cmap_name='Blues', min_shade=0.3):
    """
    Given an array-like of integer cluster labels, return a mapping
    {label: rgba_tuple} and a list of colors per element in labels
    using a sequential blue colormap.
    The mapping order is deterministic (sorted unique labels).
    """
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


def get_label_order(cluster_labels, n_shanks, n_ch_per_shank):
    unique_labels = np.unique(cluster_labels)
    #cluster_labels = np.flip(cluster_labels)
    centroid_pos = np.zeros((len(unique_labels),n_shanks))
    for shank in range(n_shanks):
        labels_shank = cluster_labels[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] 
        for j,label in enumerate(unique_labels):
            centroid_pos[j, shank] = np.mean(np.where(labels_shank == label)[0])
    return np.nanmean(centroid_pos, axis=1)


def initialize_recording(
        file_path : str | Path,
        sel_time : list
):

    try:
        import spikeinterface.full as si
        import probeinterface as pi
    except ModuleNotFoundError:
        import sys
        logger.error("Spikeinterface is not installed. Exiting")
        sys.exit()

    # initialize sinaps recording h5
    if Path(file_path).suffix == ".h5" or Path(file_path).suffix == ".bin":
        read_rec_func = si.read_sinaps_research_platform_h5 if Path(file_path).suffix == ".h5" else si.read_sinaps_research_platform
        rec = read_rec_func(file_path=file_path, stream_name="filt")
        if not rec.has_probe():
            from spikeinterface.extractors.sinapsrecordingextractors import get_sinaps_probe_info, get_sinaps_probe
            probe_type = get_sinaps_probe_info(rec)["name"]
            if probe_type == "M0004-1024p-8s_R01" or probe_type == "M0004":
                probe_type = "p1024s8"
                probe = get_sinaps_probe(probe_type)
                rec = rec.set_probe(probe)
            else:
                logger.error(f"Cannot find probe for {probe_type}. Exiting.")
                sys.exit()
    elif Path(file_path).suffix == ".cbin":
        rec = si.read_cbin_ibl(cbin_file_path=file_path, stream_name="ap")
    elif Path(file_path).suffix == ".xml":
        rec = si.read_neuroscope(file_path)
        xml_meta_path = str(Path(file_path).parent) + "/" + str(Path(file_path).stem) + ".meta"
        assert (os.path.isfile(xml_meta_path)), "Could not find associated meta file. Make sure the file ending in .meta has the same name as the .xml file and is in the same directory"
        probe = pi.neuropixels_tools.read_spikeglx(xml_meta_path)
        rec = rec.set_probe(probe)
    elif Path(file_path).suffix == ".nwb":
        try:
            rec = si.read_nwb_recording(file_path, electrical_series_path="acquisition/ElectricalSeriesLF")
        except:
            rec = si.read_nwb_recording(file_path, electrical_series_path="acquisition/ElectricalSeriesLFP")
    else:
        logger.error(f"Unrecognized file type (maybe you specified a folder instead of a file?). Getting {Path(file_path).suffix}. Expecting file ending in h5/nwb/cbin/xml.")
        raise Exception
    
    probe = rec.get_probe()
    fs = rec.get_sampling_frequency()
    n_ch = rec.get_num_channels()
    n_shanks = probe.get_shank_count()
    electrode_positions = rec.get_channel_locations().T
    rec_duration = rec.get_total_duration()
    voltage_converter = rec.get_channel_gains()[0]

    if Path(file_path).suffix == ".h5" or Path(file_path).suffix == ".bin":
        # assumes all h5 and bin recordings as SiNAPS (specs extracted from Datasheet pdf)
        if probe_type == "p1024s8":
            shank_spacing = 300
        else:
            shank_spacing = 560
        
        from spikeinterface.extractors.sinapsrecordingextractors import get_sinaps_probe_info
        probe_info = get_sinaps_probe_info(rec)
        probe_specs = {
            "x_left": 8, #static from specsheet
            "x_right": 44, #static from specsheet
            "y_top": 26, #static from specsheet
            "y_bottom": 13, #static from specsheet
            "shank_spacing": shank_spacing,
            "tip_angle": 26, #static from specsheet
            "n_cols": int(probe_info["num_cols_per_shank"]),
            "n_shanks": n_shanks,
            "n_ch_per_shank": n_ch // n_shanks
        }

        data_filtered = rec.get_traces(start_frame=sel_time[0]*fs, end_frame=sel_time[1]*fs, return_in_uV=True).T
    else:

        # taken from spec sheet in https://www.neuropixels.org/probe2-0 under shank pitch
        shank_spacing = 250

        unique_x_positions = np.unique(electrode_positions[0], axis=0)
        n_cols = len(unique_x_positions[unique_x_positions < shank_spacing])

        probe_specs = {
            "x_left": 15,
            "x_right": 38,
            "y_top": 300,
            "y_bottom": 13,
            "shank_spacing": shank_spacing,
            "tip_angle": 20,
            "n_cols": n_cols,
            "n_shanks": n_shanks,
            "n_ch_per_shank": n_ch // n_shanks
        }

        if Path(file_path).suffix == ".xml":
            rec.set_property("group", probe.shank_ids)

        if n_shanks > 1:
            rec.set_property("group", [int(x) for x in rec.get_property("group")])

            split_recording = rec.split_by("group")

            data_filtered = np.concatenate(
                [split_recording[x].get_traces(start_frame=sel_time[0]*fs, end_frame=sel_time[1]*fs).T for x in range(n_shanks)],
                axis=0
            )

            electrode_positions = np.concatenate(
                [split_recording[x].get_channel_locations().T for x in range(n_shanks)],
                axis=1
            )
        else:
            data_filtered = rec.get_traces(start_frame=sel_time[0]*fs, end_frame=sel_time[1]*fs, return_in_uV=True).T

        if Path(file_path).suffix == ".xml":
            new_order = np.lexsort((electrode_positions[1, :], electrode_positions[0, :]))

            data_filtered = data_filtered[new_order]
            electrode_positions = electrode_positions.T[new_order].T

        offset = np.min(electrode_positions[1])
        electrode_positions[1] = electrode_positions[1] - offset
        probe_specs["y_bottom"] = offset
        if n_shanks > 1:
            probe_specs["y_top"] = int(probe.probe_planar_contour.T[1].max() - electrode_positions[1].max())
        else:
            probe_specs["y_top"] = 300

    return fs, n_ch, n_shanks, electrode_positions, rec_duration, voltage_converter, data_filtered, None, probe_specs


def reshape_shank(
        array : np.ndarray,
        new_shape : int | tuple
) -> np.ndarray:
    # place first element in (0,0), second in (0,1), ..., 129th in (1,0), 130th in (1,1) and so on
    new_array = np.full(new_shape, np.nan)
    for i in range(len(array)):
        new_array[i // new_shape[1], i % new_shape[1]] = array[i]
    return new_array


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