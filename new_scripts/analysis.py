# analysis.py  ─ Section 2: Dimensionality Reduction
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster      import SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster      import KMeans
from sklearn.metrics      import silhouette_score, davies_bouldin_score
import librosa
from helper import spec_patch_vector, spec_mfcc_vector
from tqdm import tqdm 

#feature extraction


def build_feature_matrix(audio, segments, sr, extractor=spec_patch_vector):
    """Return X: ndarray [n_calls × n_features]."""
    feats = []
    for seg in segments:
        y_seg = audio[int(seg["start_s"] * sr): int(seg["end_s"] * sr)]
        feats.append(extractor(y_seg, sr))
    X = np.vstack(feats)
    print(f"Feature matrix shape: {X.shape}")   # debug print
    return X

def pca_reduce(X, n_components=0.95, *, random_state=42):
    """
    Standardise + PCA.

    Parameters
    ----------
    X : ndarray [n_samples × n_features]
    n_components : int | float
        *int*  → fixed number of PCs
        *float* in (0,1) → variance proportion
    Returns
    -------
    X_red : ndarray [n_samples × n_components_]
    pca    : fitted PCA object
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    pca    = PCA(n_components=n_components, whiten=True, random_state=random_state)
    pipe   = Pipeline([("scaler", scaler), ("pca", pca)]).fit(X)

    X_red = pipe.transform(X)
    evr   = pca.explained_variance_ratio_
    print(f"PCA retained {X_red.shape[1]} components "
          f"({evr.sum()*100:.1f} % cumulative variance)")

    return X_red, pca, scaler

def plot_pca_pairs(X_red, explained_var, max_pairs=6):
    pairs = list(combinations(range(X_red.shape[1]), 2))[:max_pairs]
    cols  = 3
    rows  = int(np.ceil(len(pairs)/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten()

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(X_red[:, i], X_red[:, j], s=10, alpha=.6)
        ax.set_xlabel(f"PC{i+1} ({explained_var[i]*100:.1f} %)")
        ax.set_ylabel(f"PC{j+1} ({explained_var[j]*100:.1f} %)")
        ax.set_title(f"PC{i+1} vs PC{j+1}")
    plt.tight_layout()
    plt.show()
    
def pipeline_pca_kmeans(feature_matrix,
                        k_range=range(2, 15),
                        var_thresh=0.95,
                        random_state=42):
    """Return best k, labels, models, silhouette trace."""
    X_red, pca, scaler = pca_reduce(feature_matrix,
                                    n_components=var_thresh,
                                    random_state=random_state)

    sil_vals = []
    km_models = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init="auto",
                    random_state=random_state).fit(X_red)
        score = silhouette_score(X_red, km.labels_)
        sil_vals.append(score)
        km_models.append(km)

    best_idx = int(np.argmax(sil_vals))
    best_k   = k_range[best_idx]
    best_km  = km_models[best_idx]

    print(f"Silhouette picked k={best_k} "
          f"(score={sil_vals[best_idx]:.3f})")

    return {
        "labels"      : best_km.labels_,
        "best_k"      : best_k,
        "silhouette"  : sil_vals,
        "pca"         : pca,
        "scaler"      : scaler,
        "kmeans"      : best_km
    }
    
def plot_pca_clusters(X_red, labels, explained_var, dims=(0, 1)):
    """Scatter PCs coloured by cluster label."""
    i, j = dims
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_red[:, i], X_red[:, j], c=labels, cmap="tab10", s=12)
    plt.xlabel(f"PC{i+1} ({explained_var[i]*100:.1f}% var)")
    plt.ylabel(f"PC{j+1} ({explained_var[j]*100:.1f}% var)")
    plt.title("PCA coloured by clusters")
    plt.colorbar(scatter, ticks=np.unique(labels))
    plt.tight_layout()
    plt.show()


# analysis.py  ─ Section 3: Spectral Clustering    
def pipeline_pca_spectral(
    feature_matrix: np.ndarray,
    *,
    k_range      = range(2, 15),
    var_thresh   = 0.95,
    affinity     = "rbf",          # 'rbf', 'nearest_neighbors', …
    random_state = 42,
):
    """PCA → Spectral Clustering; auto-select best k via silhouette."""
    X_red, pca, scaler = pca_reduce(
        feature_matrix,
        n_components = var_thresh,
        random_state = random_state,
    )

    sil_vals, models = [], []
    for k in tqdm(k_range, desc="Spectral k-loop"):
        model = SpectralClustering(
            n_clusters     = k,
            affinity       = "nearest_neighbors",   # ← sparse graph
            n_neighbors    = 15,                    # tweak 10-25
            eigen_solver   = "arpack",              # faster on <30 k samples
            assign_labels  = "kmeans",
            random_state   = random_state,
        )
        labels = model.fit_predict(X_red)

        # skip degenerate solutions (all pts same cluster)
        if len(set(labels)) < 2:
            sil_vals.append(-1)
            models.append(model)
            continue
        score = silhouette_score(X_red, labels)
        sil_vals.append(score)
        models.append(model)

    best_idx   = int(np.argmax(sil_vals))
    best_k     = k_range[best_idx]
    best_model = models[best_idx]

    print(f"Silhouette picked k={best_k} "
          f"(score={sil_vals[best_idx]:.3f})")

    return dict(
        labels      = best_model.labels_,
        best_k      = best_k,
        silhouette  = sil_vals,
        pca         = pca,
        scaler      = scaler,
        model       = best_model,
        X_red       = X_red,          # expose for plotting
    )