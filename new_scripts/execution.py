#%%
"""
execution.py
Notebook-style runner for the bird-call pipeline.
Each cell can be executed interactively.
"""

#%%
from pathlib import Path
import numpy as np
import soundfile as sf
import library as lib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#%%
# --- paths and config -------------------------------------------------
AUDIO_FILE = (Path(__file__).parent / ".." / "data" / "nestingbox1"
              / "20240927_204919" / "combined_audio.w64").resolve(strict=True)

BASE_OUT = Path(__file__).parent / "cluster_wavs"
KM_OUT   = BASE_OUT / "kmeans"
SP_OUT   = BASE_OUT / "spectral"
DB_OUT  = BASE_OUT / "dbscan"
HW_OUT  = BASE_OUT / "hier_ward"
TS_OUT  = BASE_OUT / "hier_two_step"
for d in (KM_OUT, SP_OUT, DB_OUT, HW_OUT, TS_OUT):
    d.mkdir(parents=True, exist_ok=True)

cfg = lib.Config()          # default parameters
N_NEIGHBORS = 10
print('Config loaded')

#%% helper utilities
def plot_scatter(title, labels):
    uniq = np.unique(labels)
    lab2idx = {lab: i for i, lab in enumerate(uniq)}
    colors = np.vectorize(lab2idx.get)(labels)
    cmap   = plt.cm.get_cmap('tab10', len(uniq))
    plt.figure();  plt.title(title)
    plt.scatter(X_red[:,0], X_red[:,1], c=colors, cmap=cmap, s=8)
    handles = [Line2D([],[],marker='o',linestyle='',
                      color=cmap(i), markersize=6,
                      label=("noise" if lab==-1 else f"cluster_{lab:03d}.wav"))
               for lab,i in lab2idx.items()]
    plt.legend(handles=handles, frameon=False, title="Concatenated snippet")
    plt.xlabel("PC-1"); plt.ylabel("PC-2"); plt.tight_layout(); plt.show()

print("Helper utilities ready to run the pipeline.")

#%%
# --- stream, filter, detect ------------------------------------------
all_feats   = []
all_segs    = []
broad_mat   = []

for chunk, offset in lib.iter_audio_chunks(AUDIO_FILE, cfg):
    segs = lib.detect_segments(chunk, cfg.sample_rate, cfg, offset)  # band-pass inside
    for s, e in tqdm(segs,
                     desc=f"Features @ {offset/cfg.sample_rate/3600:.1f}h",
                     leave=False,
                     unit="seg"):
        local = chunk[s - offset:e - offset]
        spec_vec  = lib.segment_spectro_db(local, cfg.sample_rate, cfg)
        broad_vec = lib.extract_broad_features_segment(local, cfg.sample_rate, cfg)
        feat_vec  = np.hstack([spec_vec, broad_vec])       # concatenate
        all_feats.append(feat_vec)
        broad_mat.append(broad_vec)

    all_segs.extend(segs)

assert all_feats, "No segments detected; adjust thresholds."
broad_mat = MinMaxScaler().fit_transform(np.vstack(broad_mat))
X = np.stack(all_feats)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"segments={len(X)}, feature-dim={X.shape[1]}")

#%%
# --- dimensionality reduction ----------------------------------------
X_red, pca = lib.reduce_dim(X, cfg)
print(f"PCA  →  {X_red.shape[1]} dimensions")

#%%
#optional for choosing k number of clusters

K_CLUSTERS = 5
print(f"{K_CLUSTERS} number of clusters is set")

#%%
# --- kmeans clustering -------------------------------------------------------
lab_km = lib.cluster_kmeans(X_red, K_CLUSTERS, cfg)
print("cluster sizes (k-means):  ", np.bincount(lab_km))
lib.save_cluster_snippets(AUDIO_FILE, cfg.sample_rate, all_segs, lab_km, KM_OUT, concat=True)

print("Done.  WAVs saved under", BASE_OUT.resolve())

#plotting ---------- PCA scatter for K-means ----------
plot_scatter("K-means Clustering", lab_km)

#%%
# --- spectral clustering -------------------------------------------------------
lab_sp = lib.cluster_spectral(X_red, K_CLUSTERS, N_NEIGHBORS, cfg)
print("cluster sizes (spectral):", np.bincount(lab_sp))
lib.save_cluster_snippets(AUDIO_FILE, cfg.sample_rate, all_segs, lab_sp, SP_OUT, concat=True)

print("Done.  WAVs saved under", BASE_OUT.resolve())

#plotting
plot_scatter("Spectral Clustering", lab_sp)

# %%
# --- DBSCAN clustering -------------------------------------------------------
best_eps, lab_db = lib.tune_dbscan(X_red, cfg)
print("cluster sizes (DBSCAN):", np.bincount(lab_db[lab_db>=0]))
lib.save_cluster_snippets(AUDIO_FILE, cfg.sample_rate, all_segs, lab_db,
                          DB_OUT, concat=True)
print("Done.  WAVs saved under", BASE_OUT.resolve())
plot_scatter(f"DBSCAN  ε={best_eps:.2f}", lab_db)

# %%# --- hierarchical clustering (Ward) ---------------------------------------
best_d, lab_hw = lib.tune_ward_distance(X_red, cfg)
lib.save_cluster_snippets(AUDIO_FILE, cfg.sample_rate, all_segs, lab_hw,
                          HW_OUT, concat=True)
plot_scatter(f"Ward d = {best_d}", lab_hw)
print("Done.  WAVs saved under", BASE_OUT.resolve())

# %%# --- two-step hierarchical clustering ---------------------------------------
# --- TWO-STEP WARD -----------------------------------------------
lab_ts = lib.two_step_ward(
    broad_mat=broad_mat,
    X_pca=X_red,
    dist_big=cfg.hier_two_big,
    dist_small=cfg.hier_two_small,
)
print("cluster sizes (2-step Ward):", np.bincount(lab_ts))
lib.save_cluster_snippets(AUDIO_FILE, cfg.sample_rate, all_segs, lab_ts,
                          TS_OUT, concat=True)
print("Done.  WAVs saved under", BASE_OUT.resolve())
plot_scatter("Two-step Hierarchical Clustering", lab_ts)

# %%
#%% Spectrogram overview (short window) ---------------------------------------
PLOTS_OUT = BASE_OUT / "plots"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

ov_spec_png, ov_wave_png = lib.save_overview_spectrogram(
    audio_path=AUDIO_FILE,
    sr=cfg.sample_rate,
    segs=all_segs,
    cfg=cfg,
    out_dir=PLOTS_OUT,
    preview_sec=15,
)
print("Saved:", ov_spec_png.resolve())
print("Saved:", ov_wave_png.resolve())

# %%
#%% Per-call spectrogram gallery (after K-means so labels appear) -------------
_ = lib.save_call_spectrograms(
    audio_path=AUDIO_FILE,
    sr=cfg.sample_rate,
    segs=all_segs,
    cfg=cfg,
    out_dir=PLOTS_OUT,
    labels=None,        # or None if you run this cell earlier
    n_preview=20,
    mode="random",        # or "longest"
)
print(f"Saved per-call spectrograms to: {PLOTS_OUT.resolve()}")

# %%
