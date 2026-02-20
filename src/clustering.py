import numpy as np
import hdbscan
import logging

from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .plotting import plot_cluster_labels, plot_dendogram, plot_dbscan


logger = logging.getLogger("lfploc")


def perform_clustering(features_reduced, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, method = 'k-means', tip_angle=26, y_top=300, y_bottom=13, custom_format="png"):
    if method == 'k-means':
        min_cluster_size = 10
        silhouette_scores = []
        if n_shanks == 1:
            cluster_range = range(3, 8) 
        else:
            cluster_range = range(4, 12)

        for k in cluster_range:
            kmeans_tmp = KMeans(n_clusters=k, random_state=0).fit(features_reduced)
            labels_tmp = kmeans_tmp.labels_.copy()
            # Mark clusters with <= min_cluster_size elements as outliers (-1)
            for cluster_id in range(k):
                if np.sum(labels_tmp == cluster_id) <= min_cluster_size:
                    labels_tmp[labels_tmp == cluster_id] = -1
            # Only compute silhouette score for non-outlier points if there are enough valid clusters
            n_valid_clusters = len(set(labels_tmp) - {-1})
            if n_valid_clusters >= cluster_range[0]:
                score = silhouette_score(features_reduced[labels_tmp != -1], labels_tmp[labels_tmp != -1])
            else:
                score = -1
            silhouette_scores.append(score)
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters based on silhouette score: {optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(features_reduced)
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

        # After clustering, create a full label array
        full_labels = kmeans.labels_
        norm = plot_cluster_labels(features_reduced, full_labels, full_labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, tip_angle=tip_angle, y_top=y_top, y_bottom=y_bottom, custom_format=custom_format)
        return full_labels, norm
    
    elif method == 'DBSCAN':
        # Use k-nearest neighbor distances to estimate a good eps value

        min_samples = max(5, features_reduced.shape[1] * 2)  # heuristic: twice the number of features, at least 5
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(features_reduced)
        distances, indices = nbrs.kneighbors(features_reduced)
        k_distances = np.sort(distances[:, -1])

        # Plot k-distance graph to visually inspect the "elbow"
        plot_dbscan(k_distances, min_samples, save_folder, custom_format=custom_format)

        # Heuristic: set eps at the value where the "elbow" occurs, or use the 90th percentile as a starting point
        eps = np.percentile(k_distances, 90)

        # Try several eps values around the heuristic and pick the one with the best silhouette score (ignore all-noise cases)
        best_score = -1
        best_labels = None
        best_eps = eps
        for eps_test in np.linspace(eps * 0.5, eps * 1.5, 10):
            dbscan = DBSCAN(eps=eps_test, min_samples=min_samples).fit(features_reduced)
            labels = dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2 or np.all(labels == -1):
                continue
            try:
                score = silhouette_score(features_reduced[labels != -1], labels[labels != -1])
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_labels = labels.copy()
                best_eps = eps_test

        if best_labels is None:
            # fallback: run with default eps
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(features_reduced)
            best_labels = dbscan.labels_
            best_eps = eps

        logger.info(f"DBSCAN: selected eps={best_eps:.3f}, min_samples={min_samples}, clusters={len(set(best_labels)) - (1 if -1 in best_labels else 0)}, silhouette={best_score:.3f}")

        full_labels = best_labels
        norm = plot_cluster_labels(features_reduced, full_labels, full_labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, custom_format=custom_format)
        return full_labels, norm
    elif method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=15)
        labels = clusterer.fit_predict(features_reduced)
        full_labels = labels
        norm = plot_cluster_labels(features_reduced, full_labels, full_labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, custom_format=custom_format)
        return full_labels, norm
    elif method == 'hierarchical':
       
        n_samples = features_reduced.shape[0]
        min_cluster_size = max(10, n_samples // 50)
        max_k = min(12, max(2, n_samples))
        best_k = None
        best_score = -np.inf

        # Find best number of clusters
        for k in range(4, max_k + 1):
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(features_reduced)
            sizes = np.bincount(labels)
            
            if np.sum(sizes < min_cluster_size) > 2:
                continue

            cv = sizes.std() / (sizes.mean() + 1e-8)
            try:
                sil = silhouette_score(features_reduced, labels)
            except Exception:
                sil = -1

            score = sil * (1.0 / (1.0 + cv))
            if score > best_score:
                best_score = score
                best_k = k

        # Fallback if no valid k found
        if best_k is None:
            best_cv = np.inf
            for k in range(2, max_k + 1):
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(features_reduced)
                sizes = np.bincount(labels)
                cv = sizes.std() / (sizes.mean() + 1e-8)
                if cv < best_cv:
                    best_cv = cv
                    best_k = k

        # Final clustering with best_k
        hierarchical = AgglomerativeClustering(n_clusters=best_k)
        labels = hierarchical.fit_predict(features_reduced)

        # Generate linkage matrix
        linked = linkage(features_reduced, method='ward')

        # Compute threshold for color split
        threshold = sorted(linked[:, 2], reverse=True)[best_k - 1]

        # Plot dendrogram with color-coded clusters and cut line
        plot_dendogram(best_k, linked, threshold, save_folder, custom_format=custom_format)

        full_labels = labels
        norm = plot_cluster_labels(features_reduced, full_labels, full_labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, custom_format=custom_format)
        return full_labels, norm

    elif method == 'None':
        full_labels = np.zeros(features_reduced.shape[0], dtype=int)
        norm = plot_cluster_labels(features_reduced, full_labels, full_labels, save_folder, n_ch_per_shank, n_cols, n_shanks, electrode_positions, n_ch, custom_format=custom_format)
        return full_labels, norm