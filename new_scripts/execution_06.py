#%%
"""
execution.py
Notebook-style runner for the bird-call pipeline.
this version: update using spline interpolation in dB domain, updated detection+warping approach
modified detection parameters(min seg length 80, merge gap 500)
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
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)
#%%
# --- paths and config -------------------------------------------------
AUDIO_FILE = (Path(__file__).parent / ".." / "data" / "nestingbox1"
              / "20240927_204919" / "combined_audio.w64").resolve(strict=True)

BASE_OUT = Path(__file__).parent / "cluster_wavs"
KM_OUT   = BASE_OUT / "kmeans"
SP_OUT   = BASE_OUT / "spectral"
HW_OUT  = BASE_OUT / "hier_ward"
for d in (KM_OUT, SP_OUT, HW_OUT):
    d.mkdir(parents=True, exist_ok=True)

cfg = lib.Config()          # default parameters
N_NEIGHBORS = 10
# --- NEW preprocessing settings ---------------------------------------------
TARGET_FRAMES = 25         # pick a fixed frame count for time-normalized patches
USE_DB = True              # interpolate in POWER, then convert to dB if True
PREPROC_MODE  = "warp"     # {"warp", "truncate"}


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

#%% --- stream, detect, CHUNK STFT (POWER) -> slice -> warp -> dB -> flatten ---
all_feats = []
all_segs  = []
lengths_frames = []

info = sf.info(str(AUDIO_FILE))
total_seconds = info.frames / info.samplerate
total_chunks = math.ceil(total_seconds / cfg.chunk_sec)


for chunk_idx, (chunk, offset) in enumerate(
        tqdm(lib.iter_audio_chunks(AUDIO_FILE, cfg, show_pbar=False),
             total=total_chunks, desc="Chunks", unit="chunk")):

    # Per-chunk step progress (3 steps: detect → STFT → slice)
    with tqdm(total=3, desc=f"Chunk {chunk_idx} steps", leave=False) as pbar:
        pbar.set_postfix_str("Detect")
        segs = lib.detect_segments(chunk, cfg.sample_rate, cfg, chunk_offset=offset, progress=False)
        pbar.update(1)

        pbar.set_postfix_str("Chunk STFT (power)")
        _, _, S_pow_chunk = lib.chunk_stft_power(
            y_chunk=chunk,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_fft=cfg.hop_fft,
            window="hann",
        )
        pbar.update(1)

        pbar.set_postfix_str("Slice")
        slices = lib.slice_segments_from_chunk_stft(
            S_pow=S_pow_chunk,
            chunk_start_sample=offset,
            segs_for_chunk=segs,
            hop_fft=cfg.hop_fft,
            n_fft=cfg.n_fft,
            progress=True,
            desc=f"Slicing @{offset/cfg.sample_rate/3600:.1f}h",
        )
        pbar.update(1)
        
    discarded_short = 0
    # Build features per slice (own progress bar)
    for item in tqdm(slices, desc=f"Feats c{chunk_idx}", leave=False, unit="slice"):
        S_slice_pow = item["S_pow"]                             # [F x t]
        # Time-normalize IN POWER
        if PREPROC_MODE == "warp":
            S_db = lib.warp_spectrogram_db_spline(
                S_slice_pow, target_frames=TARGET_FRAMES, floor_db=-120.0, use_pchip=True
            )
        elif PREPROC_MODE == "truncate":
            S_db = lib.truncate_spectrogram_db(
                S_slice_pow, target_frames=TARGET_FRAMES, floor_db=-120.0
            )
            if S_db is None:
                discarded_short += 1
                continue
        else:
            raise ValueError(f"Unknown PREPROC_MODE: {PREPROC_MODE}")
        # Flatten + duration
        patch_vec = lib.flatten_spectrogram(S_db)
        dur_s = (item["end_sample"] - item["start_sample"]) / cfg.sample_rate
        feat_vec = lib.stack_patch_with_duration(patch_vec, duration_s=dur_s)

        all_feats.append(feat_vec)
        all_segs.append((item["start_sample"], item["end_sample"]))
        lengths_frames.append(S_slice_pow.shape[1])
    if PREPROC_MODE == "truncate" and discarded_short:
        print(f"[truncate] discarded {discarded_short} segments < {TARGET_FRAMES} frames")

# Guardrail
assert all_feats, "No segments detected; adjust thresholds."

# Plot distribution of slice lengths (frames)
lengths_frames = np.asarray(lengths_frames, dtype=int)
PLOTS_OUT = BASE_OUT / "plots"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)
lib.plot_frame_length_histogram(
    lengths_frames,
    save_path=PLOTS_OUT / "slice_lengths_frames.png",
    show=True,
)

# Scale → PCA (unchanged)
X_raw = np.stack(all_feats)
X = StandardScaler().fit_transform(X_raw)
print(f"segments={len(X)}, feature-dim={X.shape[1]}")

#%%
# --- dimensionality reduction ----------------------------------------
X_red, pca = lib.reduce_dim(X, cfg)
print(f"PCA  →  {X_red.shape[1]} dimensions")

# %% Duration diagnostics & validation audio
PLOTS_OUT = BASE_OUT / "plots"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)

# 1) Duration distribution
durs_ms = lib.compute_segment_durations(all_segs, cfg.sample_rate, unit="ms")
lib.plot_duration_histogram(
    durs_ms, bins="auto", unit="ms",
    log_x=False, show=True,
    save_path=PLOTS_OUT / "duration_hist_ms.png"
)

# 2) Concatenate all calls (shortest → longest) with 100 ms gaps
VAL_OUT = BASE_OUT / "validation"
VAL_OUT.mkdir(parents=True, exist_ok=True)
preview_wav = lib.concat_calls_sorted_by_duration(
    audio_path=AUDIO_FILE,
    sr=cfg.sample_rate,
    segs=all_segs,
    out_wav=VAL_OUT / "calls_short_to_long.wav",
    gap_ms=100,
    limit=None,   # or e.g. 500 for a quick check
)
print("Wrote:", preview_wav.resolve())


#%% --- Stage-1: k-means with k=2 to separate distinct call type -------------
K_STAGE1 = 2
print(f"Stage-1 k-means: K_STAGE1 = {K_STAGE1}")

lab_stage1 = lib.cluster_kmeans(X_red, K_STAGE1, cfg)
print("Stage-1 cluster sizes (k=2):", np.bincount(lab_stage1))

plot_scatter("Stage-1 k-means (k=2)", lab_stage1)

# Per your manual validation, cluster_001 is the distinct call type
DISTINCT_CLUSTER_ID = 1
main_mask = (lab_stage1 != DISTINCT_CLUSTER_ID)

# Save Stage-1 snippets for inspection
KM_OUT_STAGE1 = KM_OUT / "stage1_k2"
KM_OUT_STAGE1.mkdir(parents=True, exist_ok=True)
lib.save_cluster_snippets(
    AUDIO_FILE, cfg.sample_rate, all_segs, lab_stage1, KM_OUT_STAGE1, concat=True
)
print("Stage-1 k=2 WAVs saved under", KM_OUT_STAGE1.resolve())

# Optional: spectra per Stage-1 cluster (uses raw flattened patches)
PLOTS_OUT = BASE_OUT / "plots"
PLOTS_OUT.mkdir(parents=True, exist_ok=True)
_ = lib.plot_cluster_spectra_from_patches(
    feats=X_raw,
    labels=lab_stage1,
    cfg=cfg,
    target_frames=TARGET_FRAMES,
    out_dir=PLOTS_OUT / "spectra_stage1_k2",
    include_duration=True,
    alpha_individual=0.10,
    lw_individual=0.4,
    lw_mean=2.0,
    show=True,
)

#%% Stage-2: clustering by duration (scalar feature only) --------------------
# Duration feature matrix: 1D (ms) → (N, 1)
X_dur = durs_ms.reshape(-1, 1)
X_dur = StandardScaler().fit_transform(X_dur)

# Optional PCA (in 1-D this is basically a no-op, but keeps the API uniform)
X_dur_red, pca_dur = lib.reduce_dim(X_dur, cfg)
print(f"Duration feature space: {X_dur_red.shape}")

# Restrict to the "main" cluster (non-distinct calls) from Stage-1
X_dur_main = X_dur_red[main_mask]
segs_main = [seg for seg, keep in zip(all_segs, main_mask) if keep]
X_raw_main = X_raw[main_mask]

# Manually choose number of duration clusters (analogous to K_CLUSTERS)
K_DUR_CLUSTERS = 3   
print(f"Stage-2 duration clustering: K_DUR_CLUSTERS = {K_DUR_CLUSTERS}")

# For now use k-means; trivially swap in spectral / DBSCAN later
lab_dur_main = lib.cluster_kmeans(X_dur_main, K_DUR_CLUSTERS, cfg)
print("Stage-2 duration cluster sizes:", np.bincount(lab_dur_main))

lab_stage2_full = np.full(len(all_segs), -1, dtype=int)   # -1 = "not clustered here"
lab_stage2_full[main_mask] = lab_dur_main

# visualize Stage-2 clusters in the SAME PCA space
plot_scatter("Stage-2 duration clustering (main cluster only)", lab_stage2_full) 

# Save snippets for duration-based clusters (main cluster only)
KM_OUT_DUR = KM_OUT / "stage2_duration"
KM_OUT_DUR.mkdir(parents=True, exist_ok=True)
lib.save_cluster_snippets(
    AUDIO_FILE, cfg.sample_rate, segs_main, lab_dur_main, KM_OUT_DUR, concat=True
)
print("Stage-2 (duration) WAVs saved under", KM_OUT_DUR.resolve())

# Spectral visualization per duration cluster, using raw patches
_ = lib.plot_cluster_spectra_from_patches(
    feats=X_raw_main,
    labels=lab_dur_main,
    cfg=cfg,
    target_frames=TARGET_FRAMES,
    out_dir=PLOTS_OUT / "spectra_stage2_duration",
    include_duration=True,
    alpha_individual=0.10,
    lw_individual=0.4,
    lw_mean=2.0,
    show=True,
)
print("Saved duration-cluster spectra under:", (PLOTS_OUT / "spectra_stage2_duration").resolve())

# %%
