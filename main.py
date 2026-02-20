import logging
import os
import time
import json
import sys
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import zscore

from src.clustering import perform_clustering
from src.dim_reduction import reduce_dimensionality
from src.plotting import plot_feature_grid_maps, plot_selected_features
from src.probe_atlas import place_probe_on_atlas
from src.utils import already_filtered, bandpass_filter, compute_psd, initialize_recording


os.environ['OMP_NUM_THREADS'] = '2'

logger = logging.getLogger("lfploc")
logger.setLevel(logging.INFO)
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setLevel(logging.INFO)
logger.addHandler(handler_stream)

DIM_REDUCTION_METHODS = ["PCA"]
CLUSTERING_METHODS = ["hierarchical", "k-means", "HDBSCAN"]


def run_analysis(
        selected_features : list,
        dim_reduction_method : str,
        clustering_method : str,
        save_dir_top : str,
        n_shanks : int,
        n_ch : int,
        sel_time : list,
        electrode_positions : list | np.ndarray,
        probe_specs : dict,
        custom_format : str
):
    
    x_left = probe_specs["x_left"]
    x_right = probe_specs["x_right"]
    y_top = probe_specs["y_top"]
    y_bottom = probe_specs["y_bottom"]
    tip_angle = probe_specs["tip_angle"]
    n_ch_per_shank = probe_specs["n_ch_per_shank"]
    n_cols = probe_specs["n_cols"]

    # remove outliers before smoothing
    for key in selected_features.keys():
        feature = selected_features[key]
        feature = feature.copy()
        window_size = 20
        for shank in range(n_shanks):
            shank_feature = feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank]
            for start in range(0, len(shank_feature), window_size):
                end = min(start + window_size, len(feature))
                window = shank_feature[start:end]
                z_scores = zscore(window, nan_policy='omit')
                outlier_mask = np.abs(z_scores) > 2.5
                window[outlier_mask] = np.nan
                shank_feature[start:end] = window
            feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] = shank_feature
        selected_features[key] = feature

    moving_avg_window = 8
    # split into shanks before smoothing and recombine after smoothing
    selected_features_smoothed = selected_features.copy()
    for key in selected_features.keys():
        feature = selected_features[key]
        smoothed_feature = np.copy(feature)
        for shank in range(n_shanks):
            shank_feature = feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank]
            smoothed_shank_feature = pd.Series(shank_feature).interpolate(method='linear', limit_direction='both').rolling(window=moving_avg_window, min_periods=1, center=True).mean()
            smoothed_feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] = smoothed_shank_feature
        selected_features_smoothed[key] = smoothed_feature

    save_folder = os.path.join(save_dir_top, f"time_{sel_time[0]}-{sel_time[1]}s")
    logger.info(f"Reading data from , time segment: {sel_time[0]}s to {sel_time[1]}s")      
    
    plot_selected_features(selected_features, n_ch_per_shank, n_cols, n_shanks, save_folder,  title='Selected Features before Smoothing', custom_format=custom_format)
    plot_selected_features(selected_features_smoothed, n_ch_per_shank, n_cols, n_shanks, save_folder, title='Selected Features after Smoothing', custom_format=custom_format)

    reduced_features = reduce_dimensionality(selected_features_smoothed, save_folder, method=dim_reduction_method, n_components=0.95, custom_format=custom_format)
    reduced_features_dict = {f'Component {i+1}': reduced_features[:, i] for i in range(reduced_features.shape[1])}
    plot_selected_features(reduced_features_dict, n_ch_per_shank, n_cols, n_shanks, save_folder, title=f'Reduced Features ({dim_reduction_method})', custom_format=custom_format)
    if n_shanks > 1:
        # Optional: save images of the interpolated grid maps
        plot_feature_grid_maps(reduced_features_dict, electrode_positions, save_folder, method='linear', title_prefix='Principal_components_map', custom_format=custom_format)

    full_labels, norm = perform_clustering(reduced_features, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, method=clustering_method, tip_angle=tip_angle, y_top=y_top, y_bottom=y_bottom, custom_format=custom_format)
    
    if n_shanks > 1:
        # Optional: save images of the interpolated grid maps
        grid_features = plot_feature_grid_maps(selected_features_smoothed, electrode_positions, save_folder, method='linear', title_prefix='Smoothed_Features', custom_format=custom_format)
        save_folder_grid = os.path.join(save_folder, "Grid_Features")
        if not os.path.exists(save_folder_grid):
            os.makedirs(save_folder_grid, exist_ok=True)
        for key in grid_features.keys():
            np.save(os.path.join(save_folder_grid, f"grid_feature_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_to_')}.npy"), grid_features[key])


    # append cluster labels to selcted_features and plot them together
    selected_features['Cluster Labels'] = full_labels
    selected_features_smoothed['Cluster Labels'] = full_labels
    plot_selected_features(selected_features, n_ch_per_shank, n_cols, n_shanks, save_folder, title=f"Relative_Power_Bands_and_Clusters", custom_format=custom_format)
    plot_selected_features(selected_features_smoothed, n_ch_per_shank, n_cols, n_shanks, save_folder, title="Relative_Power_Bands_and_Clusters_Smoothed", custom_format=custom_format)

    # save the cluster labels, electrode positions and probe parameters to a csv file
    
    probe_params = {
        'n_shanks': n_shanks,
        'n_ch_per_shank': n_ch_per_shank,
        'n_cols': n_cols,
        'x_left (um)': x_left,
        'x_right (um)': x_right,
        'y_top (um)': y_top,
        'y_bottom (um)': y_bottom,
        'tip_angle (degrees)': tip_angle
    }

    df = pd.DataFrame({
        'Channel': np.arange(n_ch),
        'Cluster Label': full_labels,
        'X (um)': electrode_positions[0],
        'Y (um)': electrode_positions[1],
        'probe_params': [probe_params]*n_ch
    })

    df.to_csv(os.path.join(save_folder, "Cluster_Labels_and_Probe_Params.csv"), index=False)
    return df, probe_params, norm, full_labels


def main(
    file_path : str | Path = None,
    csv_path : str | Path = None,
    start_time : int | float = 0,
    end_time : int | float = 20,
    custom_format : str = "png",
    ap : float = None,
    ml : float = None,
    dv : float = None
):
    
    sel_time = [start_time, end_time]

    if file_path:
        files_to_analyse = [{"file_path": file_path, "ap": ap, "ml": ml, "dv": dv}]
    elif csv_path:
        df_csv = pd.read_csv(csv_path)
        assert all(x in list(df_csv.columns) for x in ["file", "ap", "ml", "dv"]), "Got an incorrectly structured csv. Ensure columns are 'file', 'ap', 'ml', 'dv'. If a recording has no stereotactic coordinates leave blank."
        files_to_analyse = []
        for _, row in df_csv.iterrows():
            recording_path = row["file"]
            ap = row["ap"]
            ml = row["ml"]
            dv = row["dv"]
            if not Path(recording_path).suffix in [".h5", ".bin", ".dat", ".cbin", ".xml"]:
                logger.warning(f"{recording_path} has an invalid file format. Skipping.")
                continue
            files_to_analyse.append({"file_path": recording_path, "ap": ap, "ml": ml, "dv": dv})
    else:
        logger.error("file_path and csv_path undefined. Exiting.")
        return

    for file in files_to_analyse:
        
        try:

            file_path = file["file_path"]
            ap = file["ap"]
            ml = file["ml"]
            dv = file["dv"]

            # initialize recording with spikeinterface
            fs, n_ch, n_shanks, electrode_positions, rec_duration, voltage_converter, data_filtered, data_aux, probe_specs = initialize_recording(file_path, sel_time)

            # create and set save directory
            save_dir = f"output/{Path(file_path).stem}/{int(time.time())}"
            os.makedirs("output", exist_ok=True)
            os.makedirs(f"output/{Path(file_path).stem}", exist_ok=True)
            os.makedirs(save_dir)

            # save json file with input parameters
            with open(save_dir + "/input_parameters.json", "w") as f:
                parameters = {"file_path": file_path,"ap": ap, "ml": ml, "dv": dv, "time_range": sel_time}
                json.dump(parameters, f, indent=4)

            for dim_reduction_method in DIM_REDUCTION_METHODS:
                for clustering_method in CLUSTERING_METHODS:

                    save_dir_top = save_dir + f"/{dim_reduction_method}_{clustering_method}"
                    os.makedirs(save_dir_top)

                    if not already_filtered(file_path) and not Path(file_path).suffix == ".xml":
                        data_lfp = bandpass_filter(data_filtered, 1, 300, fs, order=2)
                    else:
                        data_lfp = data_filtered
                    frequencies, psd = compute_psd(data_lfp, fs)

                    # frequency bands
                    delta_band = (frequencies >= 1) & (frequencies < 4)
                    theta_band = (frequencies >= 4) & (frequencies < 8)
                    alpha_band = (frequencies >= 8) & (frequencies < 12)
                    beta_band = (frequencies >= 12) & (frequencies < 30)
                    gamma_band = (frequencies >= 30) & (frequencies < 100)
                    ripple_band = (frequencies >= 100) & (frequencies < 250)
                    total_band = (frequencies >= 1) & (frequencies < 300)

                    # set features based on psd in specific frequency bands
                    selected_features = {
                            'Delta (1 - 4 Hz)': np.sum(psd[:, delta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                            'Theta (4-8 Hz)': np.sum(psd[:, theta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                            'Alpha (8-12 Hz)': np.sum(psd[:, alpha_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                            'Beta (12-30 Hz)': np.sum(psd[:, beta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                            'Gamma (30-100 Hz)': np.sum(psd[:, gamma_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                            'Ripple (100-250)': np.sum(psd[:, ripple_band], axis=1) / np.sum(psd[:, total_band], axis=1)
                    }
                    # perform analysis
                    df, probe_params, norm, full_labels = run_analysis(selected_features, dim_reduction_method, clustering_method, save_dir_top, n_shanks, n_ch, sel_time, electrode_positions, probe_specs, custom_format)

                    if ap and ml and dv:
                        place_probe_on_atlas(df, probe_params, save_dir_top, norm, full_labels, ap, ml, dv, custom_format=custom_format)
                    else:
                        logger.warning("No AP/ML/DV coordinates provided. Skipping placement on atlas.")


        except Exception as e:
            import traceback
            logger.error(f"Analysis failed on file '{file}'")
            logger.error(traceback.format_exc())
            continue

    return


if __name__ == "__main__":
    import argparse

    example_text = '''Either --file/--csv needs to be manually specified. Defaults: start-time=0, end-time=20, custom-format=png
    
        example usage:

        python %(prog)s --file "Y:/Electrophysiology Data/my_super_cool_file.h5" --ap 1 --ml -0.5 --dv -3
        python %(prog)s --csv "C:/Users/me/my_csv_with_all_files.csv" --start-time 0 --end-time 100
    '''

    parser = argparse.ArgumentParser(epilog=example_text, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--file", "-f", type=str, dest="file_path", help="Path to file to be analysed")
    parser.add_argument("--csv", type=str, dest="csv_path", help="Path to CSV with multiple files to analyse")

    parser.add_argument("--ap", "-ap", type=float, dest="ap_coords", help="AP Coordinates. Not specifying it will skip probe placement on atlas, unless specified in excel file.")
    parser.add_argument("--ml", "-ml", type=float, dest="ml_coords", help="ML Coordinates. Not specifying it will skip probe placement on atlas, unless specified in excel file.")
    parser.add_argument("--dv", "-dv",type=float, dest="dv_coords", help="DV Coordinates. Not specifying it will skip probe placement on atlas, unless specified in excel file.")

    parser.add_argument("--start-time", type=int, dest="start_time", help="Start time of window to calculate PSD. Default 0.")
    parser.add_argument("--end-time", type=int, dest="end_time", help="End time of window to calculate PSD. Default 100.")

    parser.add_argument("--custom-format", type=str, dest="custom_format", help="Custom format to save plots. Allowed options 'png', 'eps', 'svg'")

    args = parser.parse_args()

    file_path = args.file_path
    csv_path = args.csv_path

    ap = args.ap_coords
    ml = args.ml_coords
    dv = args.dv_coords

    start_time = args.start_time
    end_time = args.end_time

    custom_format = args.custom_format

    if not start_time:
        start_time = 0
    if not end_time:
        end_time = 30
    if not custom_format:
        custom_format = "png"
    
    assert file_path or csv_path, "Neither --file or --csv was specified. Please specify at least one"
    assert custom_format in ["png", "svg", "eps"], f"Only custom formats 'png', 'svg', and 'eps' allowed. Got {custom_format}"

    if csv_path:
        assert Path(csv_path).suffix == ".csv", f"csv does not end with extension .csv. Got {Path(csv_path).suffix}"
    if file_path:
        assert Path(file_path).suffix in [".h5", ".bin", ".dat", ".cbin", ".xml", ".nwb"], f"Only recordings with extensions 'h5', 'bin', 'dat', 'cbin', 'nwb', and 'xml' supported. Got {Path(file_path).suffix}"

    main(
        file_path=file_path,
        csv_path=csv_path,
        start_time=start_time,
        end_time=end_time,
        custom_format=custom_format,
        ap=ap,
        ml=ml,
        dv=dv
    )
    