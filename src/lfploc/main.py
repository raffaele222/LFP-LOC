import numpy as np
import os
import pandas as pd
import re
import statistics
import json

from brainglobe_atlasapi import BrainGlobeAtlas
from pathlib import Path
from scipy.cluster.hierarchy import linkage
from scipy.stats import zscore
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from spikeinterface.core import BaseRecording

from .plotting import plot_cluster_labels, plot_clusters_on_probe, plot_coordinates_on_atlas, plot_dbscan, plot_dendogram, plot_evaluation_matrix, plot_feature_grid_maps, plot_pca, plot_probe_on_atlas, plot_selected_features
from .utils import bandpass_filter, build_border_image, cluster_color_map_for_labels, compute_psd, get_probe_properties, remap_labels


try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except:
    HAS_HDBSCAN = False


class Lfploc:

    def __init__(self, rec : BaseRecording):
        """
        Initialiaze an instance of the main Lfploc class.

        Parameters
        ----------
        rec : BaseRecording
            Recording initialized with SpikeInterface with probe pre-configured
        """

        self.rec = rec
        self.electrode_positions = ""
        self.df = None
        self.clustering_method = ""
        self.dimensionality_reduction_method = ""
        self.n_ch = 0

        assert self.rec.has_probe(), "Recording does not have probe attached. Exiting."

        probe = self.rec.get_probe()

        self.electrode_positions = self.rec.get_channel_locations().T
        n_shanks = probe.get_shank_count()
        n_ch = probe.get_contact_count()
        n_cols, shank_spacing = get_probe_properties(self.electrode_positions, n_shanks)

        self.probe_specs = {
            "x_left": 8,
            "x_right": 44,
            "y_top": 26,
            "y_bottom": 13,
            "shank_spacing": shank_spacing,
            "tip_angle": 26,
            "n_cols": n_cols,
            "n_shanks": n_shanks,
            "n_ch_per_shank": n_ch // n_shanks
        }

        self.n_ch = n_ch
        self.cluster_labels = None


    def run(
            self,
            feature_extraction_method : str = "PCA",
            clustering_method : str = "hierarchical",
            start_time : float = 0.0,
            end_time : float = 30.0,
            save_report_dir : Path | str = None,
            save_report_format : str = "png",
            atlas_id : str = "kim_mouse_isotropic_20um",
            ap : float = None,
            ml : float = None,
            dv : float = None
    ) -> np.ndarray:
        """
        Run main localization algorithm. If ap, ml, and dv are not passed only the cluster labels
        for each electrode are returned. If a directory is passed through the 'save_report_dir' argument
        plots generated at each step are locally saved in the format defined in 'save_report_format'.

        Parameters
        ----------
        feature_extraction_method : str = 'PCA'
            Dimensionality reduction method to run on the power spectral features
        clustering_method : str = 'hierarchical' 
            Clustering method to run after feature extraction
        start_time : float = 0.0
            Start time of recording segment to calculate power spectral features on
        end_time : float = 30.0
            End time of recording segment to calculate power spectral features on. If over length of recording
            automatically defaults to end of recording
        save_report_dir : Path | str = None
            Path to directory where to save plots generated during analysis
        save_report_format : str = 'png'
            Format in which to save each plot generated. All methods supported by matplotlib accepted
        atlas_id : str = 'kim_mouse_isotropic_20um'
            ID of atlas to use, available through brainglobe-atlasapi
        ap : str = None
            AP coordinates. If not passed placement of probe on atlas is skipped.
        ml : str = None
            ML coordinates. If not passed placement of probe on atlas is skipped.
        dv : str = None
            DV coordinates. If not passed placement of probe on atlas is skipped.

        Returns
        -------
        full_labels : np.ndarray
            (n_chs, 1) numpy array with cluster labels matching the generalized location of each electrode.
        """
        
        if save_report_dir:
            os.makedirs(save_report_dir, exist_ok=True)

        fs = self.rec.sampling_frequency
        if end_time > self.rec.get_total_duration():
            end_time = self.rec.get_total_duration()
        traces = self.rec.get_traces(start_frame=start_time*fs, end_frame=end_time*fs, return_in_uV=True).T

        psd_features = self.get_psd_features(traces)

        n_ch_per_shank = self.probe_specs["n_ch_per_shank"]
        n_shanks = self.probe_specs["n_shanks"]
        n_cols = self.probe_specs["n_cols"]

        # remove outliers before smoothing
        for key in psd_features.keys():
            feature = psd_features[key]
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
            psd_features[key] = feature

        moving_avg_window = 8
        # split into shanks before smoothing and recombine after smoothing
        psd_features_smoothed = psd_features.copy()
        for key in psd_features.keys():
            feature = psd_features[key]
            smoothed_feature = np.copy(feature)
            for shank in range(n_shanks):
                shank_feature = feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank]
                smoothed_shank_feature = pd.Series(shank_feature).interpolate(method='linear', limit_direction='both').rolling(window=moving_avg_window, min_periods=1, center=True).mean()
                smoothed_feature[shank*n_ch_per_shank:(shank+1)*n_ch_per_shank] = smoothed_shank_feature
            psd_features_smoothed[key] = smoothed_feature

        if save_report_dir:
            plot_selected_features(psd_features, n_ch_per_shank, n_cols, n_shanks, save_report_dir,  title="Selected Features before Smoothing", custom_format=save_report_format)
            plot_selected_features(psd_features_smoothed, n_ch_per_shank, n_cols, n_shanks, save_report_dir, title="Selected Features after Smoothing", custom_format=save_report_format)

        reduced_features = self.dimensionality_reduction(psd_features_smoothed, feature_extraction_method, 0.95, save_report_dir=save_report_dir, save_report_format=save_report_format)
        if save_report_dir:
            reduced_features_dict = {f'Component {i+1}': reduced_features[:, i] for i in range(reduced_features.shape[1])}
            plot_selected_features(reduced_features_dict, n_ch_per_shank, n_cols, n_shanks, save_report_dir, title=f'Reduced Features ({feature_extraction_method})', custom_format=save_report_format)
            plot_feature_grid_maps(psd_features_smoothed, self.electrode_positions, save_report_dir, method='linear', title_prefix='Smoothed_Features', custom_format=save_report_format)

        full_labels, norm = self.clustering(reduced_features, clustering_method, save_report_dir=save_report_dir, save_report_format=save_report_format)

        psd_features["Cluster Labels"] = full_labels
        psd_features_smoothed["Cluster Labels"] = full_labels
        if save_report_dir:
            plot_selected_features(psd_features, n_ch_per_shank, n_cols, n_shanks, save_report_dir, title=f"Relative_Power_Bands_and_Clusters", custom_format=save_report_format)
            plot_selected_features(psd_features_smoothed, n_ch_per_shank, n_cols, n_shanks, save_report_dir, title="Relative_Power_Bands_and_Clusters_Smoothed", custom_format=save_report_format)

        self.df = pd.DataFrame({
            "Channel": np.arange(n_shanks * n_ch_per_shank),
            "Cluster Label": full_labels,
            "X (um)": self.electrode_positions[0],
            "Y (um)": self.electrode_positions[1],
            "probe_params": [self.probe_specs]*(n_shanks*n_ch_per_shank)
        })

        if ap and ml and dv:
            if save_report_dir:
                self.place_probe_on_atlas(ap, ml, dv, save_report_dir, atlas_id, save_report_format, norm)
            else:
                print("AP, ML, DV coordinates were provided but no directory to save report in (save_report_dir). Run place_probe_on_atlas and specify a path to save the report.")

        return full_labels


    def get_psd_features(self, traces):

        # add check if recording already filtered in LFP band
        data_lfp = bandpass_filter(traces, 1, 300, self.rec.sampling_frequency, order=2)
        frequencies, psd = compute_psd(data_lfp, self.rec.sampling_frequency)

        delta_band = (frequencies >= 1) & (frequencies < 4)
        theta_band = (frequencies >= 4) & (frequencies < 8)
        alpha_band = (frequencies >= 8) & (frequencies < 12)
        beta_band = (frequencies >= 12) & (frequencies < 30)
        gamma_band = (frequencies >= 30) & (frequencies < 100)
        ripple_band = (frequencies >= 100) & (frequencies < 250)
        total_band = (frequencies >= 1) & (frequencies < 300)

        selected_features = {
                "Delta (1-4 Hz)": np.sum(psd[:, delta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                "Theta (4-8 Hz)": np.sum(psd[:, theta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                "Alpha (8-12 Hz)": np.sum(psd[:, alpha_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                "Beta (12-30 Hz)": np.sum(psd[:, beta_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                "Gamma (30-100 Hz)": np.sum(psd[:, gamma_band], axis=1) / np.sum(psd[:, total_band], axis=1),
                "Ripple (100-250 Hz)": np.sum(psd[:, ripple_band], axis=1) / np.sum(psd[:, total_band], axis=1)
        }

        return selected_features
    

    def get_run_arguments_info(self):

        return """
        'feature_extraction_method': Dimensionality reduction algorithm to run on power spectral features. Accepted: 'PCA', 'tSNE', 'UMAP'. Default 'PCA'.
        'clustering_method': Clustering algorithm to run after dim. reduction. Accepted: 'k-means', 'hierarchical', 'DBSCAN', 'HDBSCAN'. Default 'hierarchical'.
        'start_time': Start time of segment to calculate power spectral features on. Default 0.
        'end_time': End time of segment to calculate power spectral features on. Default 30.
        'save_report_dir': Directory to save a report containg every plots generated during each step. Default None (disabled).
        'save_report_format': Image format to save all plots generated into. Default 'png'.
        'atlas_id': ID of Atlas to use supported by brainglobe-atlasapi. Accepted: any atlas provided by the brainglobe-atlasapi library. Default 'kim_mouse_isotropic_20um'.
        'ap': Insertion AP coordinates in millimeters. If not defined placement of probe on atlas is skipped.
        'ml': Insertion ML coordinates in millimeters. If not defined placement of probe on atlas is skipped.
        'dv': Insertion DV coordinates in millimeters. If not defined placement of probe on atlas is skipped.
        """
    

    def clustering(self, selected_features, method, save_report_dir : str | Path = None, save_report_format : str = "png"):

        if save_report_dir:
            os.makedirs(save_report_dir, exist_ok=True)

        self.clustering_method = method

        n_shanks = self.probe_specs["n_shanks"]

        if method == 'k-means':
            
            min_cluster_size = 10
            silhouette_scores = []
            if n_shanks == 1:
                cluster_range = range(3, 8) 
            else:
                cluster_range = range(4, 12)

            for k in cluster_range:
                kmeans_tmp = KMeans(n_clusters=k, random_state=0).fit(selected_features)
                labels_tmp = kmeans_tmp.labels_.copy()
                # Mark clusters with <= min_cluster_size elements as outliers (-1)
                for cluster_id in range(k):
                    if np.sum(labels_tmp == cluster_id) <= min_cluster_size:
                        labels_tmp[labels_tmp == cluster_id] = -1
                # Only compute silhouette score for non-outlier points if there are enough valid clusters
                n_valid_clusters = len(set(labels_tmp) - {-1})
                if n_valid_clusters >= cluster_range[0]:
                    score = silhouette_score(selected_features[labels_tmp != -1], labels_tmp[labels_tmp != -1])
                else:
                    score = -1
                silhouette_scores.append(score)

            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

            kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(selected_features)
            labels = kmeans.labels_.copy()
            for cluster_id in range(optimal_k):
                if np.sum(labels == cluster_id) <= min_cluster_size:
                    labels[labels == cluster_id] = -1

            unique_labels = np.unique(labels)
            labels_new = labels.copy()
            new_id = 0
            for elements in unique_labels:
                if elements == -1:
                    continue
                labels_new[labels == elements] = new_id
                new_id += 1

            kmeans.labels_ = labels_new

            full_labels = kmeans.labels_
    
        elif method == 'DBSCAN':

            # Use k-nearest neighbor distances to estimate a good eps value
            min_samples = max(5, selected_features.shape[1] * 2)  # heuristic: twice the number of features, at least 5
            neigh = NearestNeighbors(n_neighbors=min_samples)
            nbrs = neigh.fit(selected_features)
            distances, indices = nbrs.kneighbors(selected_features)
            k_distances = np.sort(distances[:, -1])

            if save_report_dir:
                plot_dbscan(k_distances, min_samples, save_report_dir, custom_format=save_report_format)

            # Heuristic: set eps at the value where the "elbow" occurs, or use the 90th percentile as a starting point
            eps = np.percentile(k_distances, 90)

            # Try several eps values around the heuristic and pick the one with the best silhouette score (ignore all-noise cases)
            best_score = -1
            best_labels = None
            best_eps = eps
            for eps_test in np.linspace(eps * 0.5, eps * 1.5, 10):
                dbscan = DBSCAN(eps=eps_test, min_samples=min_samples).fit(selected_features)
                labels = dbscan.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2 or np.all(labels == -1):
                    continue
                try:
                    score = silhouette_score(selected_features[labels != -1], labels[labels != -1])
                except Exception:
                    score = -1
                if score > best_score:
                    best_score = score
                    best_labels = labels.copy()
                    best_eps = eps_test

            if best_labels is None:
                # fallback: run with default eps
                dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(selected_features)
                best_labels = dbscan.labels_
                best_eps = eps

            print(f"DBSCAN: selected eps={best_eps:.3f}, min_samples={min_samples}, clusters={len(set(best_labels)) - (1 if -1 in best_labels else 0)}, silhouette={best_score:.3f}")

            full_labels = best_labels

        elif method == 'HDBSCAN':

            if HAS_HDBSCAN:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
                labels = clusterer.fit_predict(selected_features)
                full_labels = labels
            else:
                print("HDBSCAN is not installed. Install with 'pip install hdbscan'")
                raise ModuleNotFoundError

        elif method == 'hierarchical':
        
            n_samples = selected_features.shape[0]
            min_cluster_size = max(10, n_samples // 50)
            max_k = min(12, max(2, n_samples))
            best_k = None
            best_score = -np.inf

            for k in range(4, max_k + 1):
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(selected_features)
                sizes = np.bincount(labels)
                
                if np.sum(sizes < min_cluster_size) > 2:
                    continue

                cv = sizes.std() / (sizes.mean() + 1e-8)
                try:
                    sil = silhouette_score(selected_features, labels)
                except Exception:
                    sil = -1

                score = sil * (1.0 / (1.0 + cv))
                if score > best_score:
                    best_score = score
                    best_k = k

            if best_k is None:
                best_cv = np.inf
                for k in range(2, max_k + 1):
                    labels = AgglomerativeClustering(n_clusters=k).fit_predict(selected_features)
                    sizes = np.bincount(labels)
                    cv = sizes.std() / (sizes.mean() + 1e-8)
                    if cv < best_cv:
                        best_cv = cv
                        best_k = k

            hierarchical = AgglomerativeClustering(n_clusters=best_k)
            labels = hierarchical.fit_predict(selected_features)

            linked = linkage(selected_features, method='ward')

            threshold = sorted(linked[:, 2], reverse=True)[best_k - 1]

            if save_report_dir:
                plot_dendogram(best_k, linked, threshold, save_report_dir, custom_format=save_report_format)

            full_labels = labels
        
        if save_report_dir:
            norm = plot_cluster_labels(
                selected_features,
                full_labels,
                full_labels,
                save_report_dir,
                self.probe_specs["n_ch_per_shank"],
                self.probe_specs["n_cols"],
                self.probe_specs["n_shanks"],
                self.electrode_positions,
                self.n_ch,
                tip_angle=self.probe_specs["tip_angle"],
                y_top=self.probe_specs["y_top"],
                y_bottom=self.probe_specs["y_bottom"], 
                custom_format=save_report_format
            )
        else:
            norm = None

        return full_labels, norm
    
    def dimensionality_reduction(self, selected_features, method, n_components, save_report_dir : str | Path = None, save_report_format : str = "png"):

        self.dimensionality_reduction_method = method

        features = np.vstack([v for v in selected_features.values()]).T

        # Define outlier threshold (e.g., any value > 2.5 is an outlier)
        outlier_mask = (np.isnan(features)).any(axis=1)

        # Store outlier indices
        outlier_indices = np.where(outlier_mask)[0]
        valid_indices = np.where(~outlier_mask)[0]

        # Exclude outliers
        features_no_outliers = features[~outlier_mask]

        # Proceed with scaling and dimensionality reduction
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_no_outliers)

        if method != "PCA":
            n_components = 3

        if method == 'PCA':
            pca = PCA(n_components=None)
            features_reduced = pca.fit_transform(features_scaled)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            n_components_95 = np.where(cumulative_variance >= n_components)[0][0] + 1 if n_components <= 1.0 else n_components
            n_components_95 = min(n_components_95, 4)
            features_reduced = features_reduced[:, :n_components_95]
            loadings = pca.components_[:n_components_95]
            # Calculate feature relevance
            feature_importance = np.sum(np.abs(loadings.T) * explained_variance[:n_components_95], axis=1)
            if save_report_dir:
                plot_pca(explained_variance, cumulative_variance, n_components_95, save_report_dir, feature_importance, selected_features, loadings, custom_format=save_report_format)
        elif method == 'tSNE':
            tsne = TSNE(n_components=n_components, random_state=0, perplexity=30)
            features_reduced = tsne.fit_transform(features_scaled)
        elif method == 'UMAP':
            if HAS_UMAP:
                reducer = umap.UMAP(n_components=n_components, random_state=0)
                features_reduced = reducer.fit_transform(features_scaled)
            else:
                print("UMAP is not installed. Install with 'pip install umap-learn'")
                raise ModuleNotFoundError

        return features_reduced
    

    def place_coordinates_on_atlas(self, ap : float, ml : float, dv : float, save_report_dir : str | Path, atlas_id : str = "kim_mouse_isotropic_20um", save_report_format : str = "png", borders : bool = False):

        os.makedirs(save_report_dir, exist_ok=True)

        # Bregma position in atlas (in micron)
        bregma_pos = [5400, 0, 0]

        # Load the Allen Mouse Brain Atlas
        atlas = BrainGlobeAtlas(atlas_id)

        # get resolution from atlas_id
        # Extract resolution from atlas_id
        match = re.search(r'(\d+)\s*um', atlas_id)
        if match:
            resolution_um = int(match.group(1))
        else:
            resolution_um = None
        print(f"Atlas resolution: {resolution_um} um")

        # load structure information using json file
        json_path = os.path.join(atlas.root_dir, "structures.json")
        with open(json_path, "r") as f:
            structures = json.load(f)

        # Build a lookup: region_id -> RGB color
        id_to_rgb = {}
        all_rgb = []
        for region in structures:
            region_id = region['id']
            rgb = region.get('rgb_triplet', [255, 255, 255])  # fallback to white
            id_to_rgb[region_id] = rgb
            all_rgb.append(rgb)

        # annotation image
        annotation_image = atlas.annotation
        annotation_shape = annotation_image.shape

        # position of text defining stereotactic coordinated
        text_pos = (int(0.12*annotation_shape[1]), int(0.72*annotation_shape[0]))
            
        # Convert AP, ML, DV (mm) to µm
        AP_um = ap * 1000
        ML_um = ml * 1000
        DV_um = dv * 1000

        # Convert AP, ML, DV (µm) to atlas voxel indices
        # AP: z axis, ML: x axis, DV: y axis
        slice_z = int((bregma_pos[0] - AP_um )/ resolution_um)
        slice_y = int((bregma_pos[1] + DV_um) / resolution_um)
        slice_x = int((bregma_pos[2] - ML_um) / resolution_um)

        selected_atlas = atlas.annotation
        selected_rgb_map = id_to_rgb

        # Prepare the annotation slice
        selected_annotated_image = selected_atlas[slice_z, :, :]

        # Map annotation IDs either to their native region colors or to border-only view.
        rgb_image = np.zeros((*selected_annotated_image.shape, 3), dtype=np.uint8)

        # get offsets based on atlas
        x_atlas_offset = annotation_shape[2]//2
        y_atlas_offset = rgb_image.shape[2]

        if borders:
            rgb_image = build_border_image(selected_annotated_image)
        else:
            for region_id, rgb in selected_rgb_map.items():
                mask = selected_annotated_image == region_id
                rgb_image[mask] = rgb

        # Add distance between border of image and first cortical layer to DV
        # get first channel of RGB to simplify method
        rgb_image_first = rgb_image[:, :, 0]
        # get only values on the y axis aligned to the first point on the x axis in the probe
        values_y = rgb_image_first[:, int(slice_x + x_atlas_offset)]
        # find first index where the image on the y-axis isn't equal to 0 (should lead to start of cortex)
        idx_cortex = np.argmax(values_y != values_y[0])
        # subtract this offset from DV and probe_y_px coordinates
        offset_DV_um = idx_cortex * resolution_um
        DV_um = DV_um - offset_DV_um
        dv = DV_um / 1000
        print(f"Calculated offset between border and cortex: {offset_DV_um}um")

        x_plot = slice_x + x_atlas_offset
        y_plot = y_atlas_offset - slice_y + idx_cortex
        plot_coordinates_on_atlas(ap, ml, dv, rgb_image, x_plot, y_plot, text_pos, resolution_um, borders, save_report_dir, save_report_format)
    

    def place_probe_on_atlas(self, ap : float, ml : float, dv : float,  save_report_dir : str | Path, atlas_id : str = "kim_mouse_isotropic_20um", save_report_format : str = "png", norm = None):

        if not isinstance(self.df, pd.DataFrame):
            print("Labelling algorithm not run. Find labels with run()")
            return

        # Bregma position in atlas (in micron)
        bregma_pos = [5400, 0, 0]

        # Load the Allen Mouse Brain Atlas
        atlas = BrainGlobeAtlas(atlas_id)

        # Load relabelled atlas where similar regions are plotted with the same color
        relabeled_atlas = np.load("./lib/relabeled_atlas.npy")
        rgb_map_relabelled = np.load("./lib/rgb_map.npy", allow_pickle=True).item()

        # get resolution from atlas_id
        match = re.search(r'(\d+)\s*um', atlas_id)
        if match:
            resolution_um = int(match.group(1))
            print(f"Atlas resolution: {resolution_um} um")
        else:
            resolution_um = None
            print("Resolution not found in atlas_id.")

        # load structure information using json file
        json_path = os.path.join(atlas.root_dir, "structures.json")
        with open(json_path, "r") as f:
            structures = json.load(f)

        # Build a lookup: region_id -> RGB color
        id_to_rgb = {}
        all_rgb = []
        for region in structures:
            region_id = region['id']
            rgb = region.get('rgb_triplet', [255, 255, 255])  # fallback to white
            id_to_rgb[region_id] = rgb
            all_rgb.append(rgb)

        ## add warning no labels found in the atlas

        # reference image
        reference_image = atlas.reference

        # annotation image
        annotation_image = atlas.annotation
        annotation_shape = annotation_image.shape

        # convert AP, ML, DV (mm) to µm
        AP_um = ap * 1000
        ML_um = ml * 1000
        DV_um = dv * 1000

        # Convert AP, ML, DV (µm) to atlas voxel indices
        # AP: z axis, ML: x axis, DV: y axis
        slice_z = int((bregma_pos[0] - AP_um )/ resolution_um)
        slice_y = int((bregma_pos[1] + DV_um) / resolution_um)
        slice_x = int((bregma_pos[2] - ML_um) / resolution_um)

        if not atlas_id == "kim_mouse_isotropic_20um" or self.probe_specs["n_shanks"] == 1: 
            selected_atlas = atlas.annotation
            selected_rgb_map = id_to_rgb
        else:
            selected_atlas = relabeled_atlas
            selected_rgb_map = rgb_map_relabelled


        # Prepare the annotation slice
        selected_annotated_image = selected_atlas[slice_z, :, :]
        selected_reference_image = reference_image[slice_z, :,:]

        # Map annotation IDs to RGB colors
        rgb_image = np.zeros((*selected_annotated_image.shape, 3), dtype=np.uint8)

        for region_id, rgb in selected_rgb_map.items():
            mask = selected_annotated_image == region_id
            rgb_image[mask] = rgb

        # Add distance between border of image and first cortical layer to DV
        # get first channel of RGB to simplify method
        rgb_image_first = rgb_image[:, :, 0]
        # get only values on the y axis aligned to the first point on the x axis in the probe
        values_y = rgb_image_first[:, int(slice_x + annotation_shape[2]//2)]
        # find first index where the image on the y-axis isn't equal to 0 (should lead to start of cortex)
        idx_cortex = np.argmax(values_y != values_y[0])
        # subtract this offset from DV and probe_y_px coordinates
        offset_DV_um = idx_cortex * resolution_um
        DV_um = DV_um - offset_DV_um
        dv = DV_um / 1000
        print(f"Calculated offset between border and cortex: {offset_DV_um}um")

        save_path = save_report_dir + "/atlas"
        os.makedirs(save_path, exist_ok=True)

        probe_x_um = self.df['X (um)'].values
        probe_y_um = self.df['Y (um)'].values

        # calculate offsets in the x and y directions
        x_offset = annotation_shape[2]//2 # center at the midline of the brain
        y_offset = int(np.max(probe_y_um)/resolution_um)

        # flip y_values to match atlas direction
        probe_y_um = np.flip(probe_y_um)
        cluster_labels = self.df['Cluster Label'].values

        # Convert probe coordinates (um) to atlas pixel coordinates
        probe_x_px =  x_offset + slice_x + (probe_x_um / resolution_um)
        probe_y_px = rgb_image.shape[2] - y_offset - slice_y + idx_cortex + (probe_y_um / resolution_um)

        # Create blue shades mapping going darker from top clusters to bottom clusters
        cluster_labels, sorted_labels = remap_labels(cluster_labels, self.probe_specs)
        unique_labels = np.unique(cluster_labels)
        cluster_to_color, colors = cluster_color_map_for_labels(cluster_labels, cmap_name='Blues', min_shade=0.3)

        plot_clusters_on_probe(cluster_labels, cluster_to_color, self.probe_specs, save_report_dir, save_report_format)

        # Plot anatomy and probe clusters
        text_pos = (int(0.12*annotation_shape[1]), int(0.72*annotation_shape[0])) 
        plot_probe_on_atlas(rgb_image, colors, norm, probe_x_px, probe_y_px, self.probe_specs, np.round(ap,1), np.round(ml,1), np.round(dv,1), os.path.join(save_path, f'probe_labelled_atlas_original_pos.{save_report_format}'), resolution_um, text_pos, custom_format=save_report_format)
        plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px, probe_y_px, self.probe_specs, np.round(ap), np.round(ml,1), np.round(dv,1), os.path.join(save_path, f'probe_reference_atlas_original_pos.{save_report_format}'), resolution_um, text_pos, custom_format=save_report_format)

        # REALIGNMENT
        # find the best match in the x, y and z position to minimize the variation within clusters
        n_pixel_region = np.zeros(len(unique_labels))
        label_regions = np.zeros(len(unique_labels))
        x_shift = np.arange(-30, 30) # x shift range in pixels
        y_shift = np.arange(-30, 30) # y shift range in pixels 
        z_shift = np.arange(-8, 8) # z shift on the coronal plane
        shape = [len(x_shift), len(y_shift), len(z_shift)]

        evaluation_mat_label = np.zeros(shape) # evaluation matrix to check where the cluster match best the anatomy

        annotated_image = selected_atlas[slice_z , :, :]

        # Assign to each region the mode value of where it was originally placed
        for k, label in enumerate(unique_labels):
            label_x_px = probe_x_px[cluster_labels == label].astype(np.int16) 
            label_y_px = probe_y_px[cluster_labels == label].astype(np.int16)
            values_label = annotated_image[label_y_px, label_x_px]
            label_regions[k] = statistics.mode(values_label)

        for m, z_s in enumerate(z_shift):
            selected_reference_image = reference_image[slice_z + z_s, :,:]
            annotated_image = selected_atlas[slice_z + z_s, :, :]
            for i, x_s in enumerate(x_shift):
                for j, y_s in enumerate(y_shift):
                    for k, label in enumerate(unique_labels):
                        # if there is only one shank, only consider shifts that keep the probe in the same hemisphere
                        if self.probe_specs["n_shanks"] == 1:
                            if (probe_x_px[cluster_labels == label].mean() < annotation_shape[2]//2 and probe_x_px[cluster_labels == label].mean() + x_s > annotation_shape[2]//2) or (probe_x_px[cluster_labels == label].mean() > annotation_shape[2]//2 and probe_x_px[cluster_labels == label].mean() + x_s < annotation_shape[2]//2):
                                continue
                        label_x_px = probe_x_px[cluster_labels == label].astype(np.int16) + x_s
                        label_y_px = probe_y_px[cluster_labels == label].astype(np.int16) + y_s
                        values_label = annotated_image[label_y_px, label_x_px]

                        n_pixel_region[k] = np.sum(values_label==label_regions[k] )

                    evaluation_mat_label[i, j, m] = np.sum(n_pixel_region)

        # Find the indices of the minimum value in the evaluation matrix
        max_ind = np.unravel_index(np.argmax(evaluation_mat_label), evaluation_mat_label.shape)
        best_x_shift = x_shift[max_ind[0]]
        best_y_shift = y_shift[max_ind[1]]
        best_z_shift = z_shift[max_ind[2]]

        print("Best shift:", best_x_shift * resolution_um, best_y_shift * resolution_um, best_z_shift * resolution_um)
        print("N clusters in appropriate region = ",evaluation_mat_label[max_ind[0], max_ind[1], max_ind[2]] )

        plot_extent = [x_shift[0]*resolution_um, x_shift[-1]*resolution_um, y_shift[0]*resolution_um, y_shift[-1]*resolution_um]

        plot_evaluation_matrix(evaluation_mat_label, max_ind, z_shift, best_x_shift, best_y_shift, resolution_um, plot_extent, save_report_dir, save_report_format)

        selected_annotated_image = selected_atlas[slice_z + best_z_shift, :, :]

        # Map annotation IDs to RGB colors
        rgb_image_sel = np.zeros((*selected_annotated_image.shape, 3), dtype=np.uint8)

        for region_id, rgb in selected_rgb_map.items():
            mask = selected_annotated_image == region_id
            rgb_image_sel[mask] = rgb

        selected_reference_image = reference_image[slice_z + best_z_shift,:,:]

        # Plot anatomy and probe clusters in adjusted position
        AP_new = np.round(ap - best_z_shift*resolution_um/1000,1)
        ML_new = np.round(ml - best_x_shift * resolution_um/1000, 1)
        DV_new = np.round(dv - best_y_shift*resolution_um/1000, 1)

        plot_probe_on_atlas(selected_reference_image, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, self.probe_specs, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_reference_atlas_adjusted.{save_report_format}'), resolution_um, text_pos, custom_format=save_report_format)
        plot_probe_on_atlas(rgb_image_sel, colors, norm, probe_x_px + best_x_shift, probe_y_px + best_y_shift, self.probe_specs, AP_new, ML_new, DV_new, os.path.join(save_path, f'probe_original_label_atlas_adjusted.{save_report_format}'), resolution_um, text_pos, custom_format=save_report_format)