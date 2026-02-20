import numpy as np
import umap

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from .plotting import plot_pca


def reduce_dimensionality(selected_features, save_folder, method='PCA', n_components=3, custom_format="png"):
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

        plot_pca(explained_variance, cumulative_variance, n_components_95, save_folder, feature_importance, selected_features, loadings, custom_format=custom_format)
        return features_reduced
    elif method == 'tSNE':
        tsne = TSNE(n_components=n_components, random_state=0, perplexity=30)
        features_reduced = tsne.fit_transform(features_scaled)
        return features_reduced
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=0)
        features_reduced = reducer.fit_transform(features_scaled)
        return features_reduced

    else:
        raise ValueError("Method must be 'PCA' or 'tSNE'")