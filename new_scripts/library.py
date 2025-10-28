"""
library.py
signal-processing and clustering utilities for a 24 h nesting-box
recording.  Designed for memory-constrained laptops and guided by best practice
in bio-acoustics: band-pass 2-8 kHz, RMS activity detection, power-domain
averaging, log-dB scaling, and Parseval-consistent operations.

Author: Pigbythesea
Created: 2025-08-05
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft, find_peaks, resample_poly
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Dict, Any, Iterable, List, Tuple
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
# Configuration dataclass
# -----------------------------------------------------------------------------#

@dataclass
class Config:
    """All tunables live here so `execution.py` can override as needed."""
    sample_rate: int = 48_000                    # Hz
    bandpass_hz: Tuple[int, int] = (1_000, 12_000)
    chunk_sec: int = 7200                        # 2h of audio per segment
    rms_win_ms: int = 25                         # RMS frame length
    rms_hop_ms: int = 10                         # RMS hop
    sigma_mult: float = 3.0                      # baseline + sigma multiplier for peak
    min_seg_ms: int = 50                         # discard shorter segments
    merge_gap_ms: int = 30                       # merge close bursts
    stft_win: int = 1024
    stft_hop: int = 512
    pca_variance: float = 0.95                   # keep 95 % variance
    random_state: int = 42
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 5
    first_k: int = 30     # two-step stage-1
    final_k: int = 6      # two-step stage-2
    hier_n_clusters: int | None = None      # Ward single-stage
    hier_two_big: int = 0.35             # coarse cut
    hier_two_small: int = 0.12            # per-big-cluster k-means
    dbscan_grid      : tuple = (0.6, 0.8, 1.0, 1.2, 1.4)  # candidate eps
    hier_grid        : tuple = tuple(range(50, 401, 25))  # candidate distance thresholds
    hier_penalty     : float = 1e-1                       # as in reference.py
    # --- spectrogram parameters (shared everywhere) --------------------------
    n_fft: int  = 1024         # window length in samples
    hop_fft: int = 512         # step size  (= 50 % overlap)
    ref_db_floor: float = 1e-12  # floor to avoid -inf in log10




# -----------------------------------------------------------------------------#
# I/O – memory-safe chunk iterator
# -----------------------------------------------------------------------------#

def iter_audio_chunks(path: Path, cfg: Config, show_pbar: bool = True):
    sr = cfg.sample_rate
    frames_per_chunk = cfg.chunk_sec * sr
    with sf.SoundFile(path) as snd:
        total_chunks = int(np.ceil(snd.frames / frames_per_chunk))
        pbar = tqdm(total=total_chunks, desc="Streaming audio", unit="chunk") if show_pbar else None
        if snd.samplerate != sr:
            raise ValueError(f"Expected {sr} Hz but file is {snd.samplerate} Hz")
        idx = 0
        while True:
            data = snd.read(frames_per_chunk, dtype='float32', always_2d=True)
            if data.size == 0:
                break
            y = data.mean(axis=1)  # mono
            yield y, idx
            idx += len(y)
            if pbar: pbar.update(1)
        if pbar: pbar.close()

# -----------------------------------------------------------------------------#
# Filtering
# -----------------------------------------------------------------------------#

def bandpass(y: np.ndarray, sr: int,
             low: float, high: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    sos = butter(order, [low, high], btype='bandpass',
                 fs=sr, output='sos')
    return sosfiltfilt(sos, y)


# -----------------------------------------------------------------------------#
# RMS envelope and segmentation
# -----------------------------------------------------------------------------#

def rms_envelope(y: np.ndarray, sr: int, win_samp: int, hop_samp: int) -> np.ndarray:
    """Vectorised RMS over sliding window."""
    # Pad to multiple of hop for convenience
    pad = (-len(y)) % hop_samp
    if pad:
        y = np.pad(y, (0, pad), mode='constant')
    y_frames = y.reshape(-1, hop_samp)
    # Compute frame energies via stride trick
    frame_energy = np.square(y_frames).mean(axis=1)
    win = np.ones(win_samp // hop_samp)
    rms = np.sqrt(np.convolve(frame_energy, win, 'same'))
    return rms


def detect_segments(y: np.ndarray,
                    sr: int,
                    cfg: Config,
                    progress: bool = False,
                    chunk_offset: int = 0) -> List[Tuple[int, int]]:
    """
    Return list of (start_sample, end_sample) pairs **in global sample units**
    for vocal-activity segments detected within `y`.
    """
    # bandpass the audio before detecting segments
    y = bandpass(y, sr, *cfg.bandpass_hz)
    # Parameters in samples
    hop = int(cfg.rms_hop_ms * 1e-3 * sr)
    win = int(cfg.rms_win_ms * 1e-3 * sr)
    env = rms_envelope(y, sr, win, hop)
    
    # convert RMS-amplitude: linear power
    power_env = np.square(env)  # linear power
    peaks, _ = find_peaks(power_env,
                      height=np.percentile(power_env, 80),
                      distance=int(cfg.min_seg_ms / cfg.rms_hop_ms))

    baseline_pool = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        slice_ = power_env[a:b]
        baseline_pool.extend(slice_[slice_ < np.median(slice_)])

    if not baseline_pool:                      # fallback if no gaps found
        baseline_pool = power_env

    mu  = float(np.mean(baseline_pool))
    sig = float(np.std(baseline_pool, ddof=0))
    thresh_power = mu + 3.0 * sig              # reference: μ+3σ

    active = power_env > thresh_power
    iterable = tqdm(active, desc="Detect segments", leave=False, unit="frame") if progress else active

    # Frame indices → sample indices
    segs = []
    in_seg = False
    seg_start_frame = 0
    for i, a in enumerate(iterable):
        if a and not in_seg:
            in_seg = True
            seg_start_frame = i
        elif not a and in_seg:
            in_seg = False
            segs.append((seg_start_frame, i))

    # Post processing (merge & min length)
    merged: List[Tuple[int, int]] = []
    gap_frames = int(cfg.merge_gap_ms / cfg.rms_hop_ms)
    min_frames = int(cfg.min_seg_ms / cfg.rms_hop_ms)

    for s, e in segs:
        if (e - s) < min_frames:
            continue
        if not merged:
            merged.append([s, e])
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e <= gap_frames:
                merged[-1][1] = e
            else:
                merged.append([s, e])

    # Convert to sample offsets (global)
    seg_sample: List[Tuple[int, int]] = []
    for s_f, e_f in merged:
        start = chunk_offset + s_f * hop
        end = chunk_offset + e_f * hop
        seg_sample.append((start, end))

    return seg_sample


# -----------------------------------------------------------------------------#
# Spectrogram & feature extraction
# -----------------------------------------------------------------------------#

def segment_mag_linear(y_seg: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """
    Produce a *frequency-power* vector (time-averaged dB spectrum) for a segment.
    Dimensionality = n_freq_bins determined by STFT parameters.
    """
    f, t, Z = stft(
        y_seg,
        fs=sr,
        nperseg=cfg.stft_win,
        noverlap=cfg.stft_win - cfg.stft_hop,
        window="hann",
        padded=False,
        boundary=None,
    )
    power = np.abs(Z)**2  # linear power
    mag = np.abs(Z).astype(np.float32)
    avg_mag = mag.mean(axis=1)
    return avg_mag

def segment_spectro_db(y_seg: np.ndarray, sr: int, cfg) -> np.ndarray:
    """
    Return a 1-D feature vector: mean log-power spectrum of one segment.
    Length = n_fft//2 + 1.
    """
    _, _, Z = stft(
        y_seg,
        fs=sr,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_fft,
        window="hann",
        padded=False,
        boundary=None,
    )
    power   = np.square(np.abs(Z), dtype=np.float32)          # linear power
    avg_pow = power.mean(axis=1)                              # **pool in linear domain**
    feat_db = 10.0 * np.log10(avg_pow + cfg.ref_db_floor)     # convert after pooling
    return feat_db.astype(np.float32)

def extract_broad_features_segment(y_seg: np.ndarray, sr: int, cfg) -> np.ndarray:
    """
    Return [duration_s, total_energy, spectral_centroid_Hz, bandwidth_Hz].

    • duration   – seconds (linear domain)
    • energy     – ∑(y²)  (linear power, Parseval-consistent)
    • centroid   – Σ f·P(f) / Σ P(f)  (Hz, linear power weight)
    • bandwidth  – √(Σ (f-centroid)² · P(f) / Σ P(f))  (Hz)
    """
    dur = len(y_seg) / sr

    # FFT once for centroid + bandwidth
    freqs, _, Z = stft(
        y_seg,
        fs=sr,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_fft,
        window="hann",
        padded=False,
        boundary=None,
    )
    P = np.square(np.abs(Z), dtype=np.float32)     # linear power
    P_mean = P.mean(axis=1)                        # average over time

    energy = float(P_mean.sum())
    if energy == 0:                                # silence guard
        return np.array([dur, 0.0, 0.0, 0.0], dtype=np.float32)

    centroid = float(np.sum(freqs * P_mean) / energy)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * P_mean) / energy))
    return np.array([dur, energy, centroid, bandwidth], dtype=np.float32)

def two_step_ward(
    broad_mat: np.ndarray,
    X_pca: np.ndarray,
    dist_big: float,
    dist_small: float,
) -> np.ndarray:
    """
    Two-stage Ward that mirrors reference.py:
    1) coarse cut on *broad features*
    2) refine each coarse group on *PCA scores*

    Returns 1-D label array (length = n_segments).
    """
    coarse = ward_by_distance(broad_mat, dist_big)     # step-1

    final = np.zeros_like(coarse)
    next_lbl = 0
    for cid in np.unique(coarse):
        idx = np.where(coarse == cid)[0]               # indices of this macro cluster
        sub_labels = ward_by_distance(X_pca[idx], dist_small)
        final[idx] = sub_labels + next_lbl
        next_lbl += sub_labels.max() + 1               # ensure global uniqueness
    return final.astype(int)


# -----------------------------------------------------------------------------#
# NEW: Per-segment spectrogram (fixed resolution), time-warp, and utilities
# -----------------------------------------------------------------------------#

def segment_spectrogram_power(
    y_seg: np.ndarray,
    sr: int,
    n_fft: int,
    hop_fft: int,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute a per-segment STFT **linear power** spectrogram with fixed time–freq
    resolution defined by (n_fft, hop_fft). No padding; boundary=None.

    Parameters
    ----------
    y_seg : 1-D ndarray
        Mono audio samples for a single detected segment.
    sr : int
        Sampling rate in Hz.
    n_fft : int
        STFT window length (samples).
    hop_fft : int
        STFT hop size (samples).
    window : str
        STFT window type (default: 'hann').

    Returns
    -------
    S_pow : 2-D ndarray, shape (n_freq_bins, n_frames), dtype float32
        Linear power spectrogram (|STFT|^2). Frequencies unchanged; frames vary
        with segment duration.
    """
    # Compute STFT without padding
    f, t, Z = stft(
        y_seg,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_fft,
        window=window,
        padded=False,
        boundary=None,
    )
    S_pow = (np.abs(Z) ** 2).astype(np.float32)
    return S_pow


def power_to_db(S_power: np.ndarray, ref_floor: float = 1e-12) -> np.ndarray:
    """
    Convert linear power to log10 dB with a small floor to avoid -inf.

    Parameters
    ----------
    S_power : 2-D ndarray
        Linear power spectrogram.
    ref_floor : float
        Additive floor before log10.

    Returns
    -------
    S_db : 2-D ndarray, same shape as S_power, dtype float32
        Log10 power in dB.
    """
    S_db = 10.0 * np.log10(S_power + float(ref_floor))
    return S_db.astype(np.float32)


def time_interpolate_spectrogram(
    S: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """
    Warping: Interpolate a spectrogram **along the time axis only** to produce
    exactly `target_frames` frames, leaving frequency bins unchanged.

    Notes
    -----
    • Works on either linear-power or dB input; for most faithful energy
      relationships, prefer interpolating **linear power**, then convert to dB.
    • Uses 1-D linear interpolation per frequency bin.

    Parameters
    ----------
    S : 2-D ndarray, shape (n_freq_bins, n_frames)
        Input spectrogram (power or dB).
    target_frames : int
        Desired number of time frames.

    Returns
    -------
    S_warp : 2-D ndarray, shape (n_freq_bins, target_frames)
        Time-warped spectrogram in the same domain (power or dB) as input.
    """
    F, T = S.shape
    if T == target_frames:
        return S.copy()
    if T <= 0:
        return np.zeros((F, target_frames), dtype=S.dtype)
    if target_frames <= 0:
        raise ValueError("target_frames must be a positive integer")

    x_old = np.linspace(0.0, 1.0, T, endpoint=True, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_frames, endpoint=True, dtype=np.float32)

    S_warp = np.empty((F, target_frames), dtype=S.dtype)
    # Interpolate each frequency bin independently
    for i in range(F):
        S_warp[i] = np.interp(x_new, x_old, S[i])
    return S_warp


def duration_seconds(y_seg: np.ndarray, sr: int) -> float:
    """
    Duration of a segment in seconds.
    """
    return float(len(y_seg)) / float(sr)


def flatten_spectrogram(S: np.ndarray, order: str = "C") -> np.ndarray:
    """
    Flatten a 2-D spectrogram to 1-D.
    Parameters
    ----------
    S : 2-D ndarray
        Spectrogram (power or dB).
    order : {'C','F'}
        Memory order for flattening (default 'C').

    Returns
    -------
    v : 1-D ndarray, dtype float32
    """
    return np.asarray(S, dtype=np.float32).reshape(-1, order=order)


def stack_patch_with_duration(patch_vec: np.ndarray, duration_s: float) -> np.ndarray:
    """
    Concatenate a flattened spectrogram patch with a scalar duration feature.
    Returns a new 1-D float32 vector [patch_vec ; duration_s].
    """
    dur = np.array([float(duration_s)], dtype=np.float32)
    return np.concatenate([patch_vec.astype(np.float32, copy=False), dur], axis=0)

# -----------------------------------------------------------------------------#
# Chunk-level STFT (POWER) and slicing
# -----------------------------------------------------------------------------#

def chunk_stft_power(y_chunk: np.ndarray,
                     sr: int,
                     n_fft: int,
                     hop_fft: int,
                     window: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT on an entire streamed chunk. Returns linear POWER spectrogram.

    Returns
    -------
    f : ndarray [Hz]
    t : ndarray [s]             (relative to chunk start)
    S_pow : float32 [F x T]     (|STFT|^2)
    """
    f, t, Z = stft(
        y_chunk,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_fft,
        window=window,
        padded=False,
        boundary=None,
    )
    S_pow = (np.abs(Z) ** 2).astype(np.float32)
    return f, t, S_pow


def slice_segments_from_chunk_stft(S_pow: np.ndarray,
                                   chunk_start_sample: int,
                                   segs_for_chunk: list[tuple[int, int]],
                                   hop_fft: int,
                                   n_fft: int,
                                   progress: bool = True,
                                   desc: str | None = None) -> list[dict]:
    """
    Slice a CHUNK power spectrogram by (start,end) segment times (global samples).

    Returns a list of dicts with:
      {
        "S_pow": [F x t_local],
        "start_frame": int, "end_frame": int,
        "start_sample": int, "end_sample": int,
        "chunk_start_sample": int,
      }
    """
    T = S_pow.shape[1]
    out = []
    if progress: 
        iterable = tqdm(segs_for_chunk, desc=desc or "Slicing segments", leave=False, unit="segment")
    
    for (s_glob, e_glob) in iterable if progress else segs_for_chunk:
        # sample indices relative to this chunk
        s_rel = s_glob - chunk_start_sample
        e_rel = e_glob - chunk_start_sample

        # skip if fully outside
        if e_rel <= 0 or s_rel >= (T * hop_fft + n_fft):
            continue

        # map samples -> STFT frame grid k*hop
        s_frame = int(np.floor(max(0, s_rel) / hop_fft))
        e_frame = int(np.ceil(max(0, e_rel) / hop_fft))

        # clamp; ensure at least one column
        s_frame = max(0, min(T - 1, s_frame))
        e_frame = max(s_frame + 1, min(T, e_frame))

        S_slice = S_pow[:, s_frame:e_frame].copy()
        out.append({
            "S_pow": S_slice,
            "start_frame": s_frame,
            "end_frame": e_frame,
            "start_sample": s_glob,
            "end_sample": e_glob,
            "chunk_start_sample": chunk_start_sample,
        })
    return out


def plot_frame_length_histogram(lengths_frames: np.ndarray,
                                save_path: Optional[Path] = None,
                                show: bool = True,
                                title: str = "Distribution of call lengths (STFT frames)"):
    if lengths_frames.size == 0:
        print("No slices to plot.")
        return None
    n = lengths_frames.size
    n_bins = int(np.clip(np.sqrt(n), 10, 60))
    med = float(np.median(lengths_frames))
    mean = float(np.mean(lengths_frames))

    plt.figure(figsize=(7, 3.2))
    plt.hist(lengths_frames, bins=n_bins, edgecolor="black", linewidth=0.5)
    plt.axvline(med,  linestyle="--", linewidth=1.2, label=f"median = {med:.1f} frames")
    plt.axvline(mean, linestyle=":",  linewidth=1.2, label=f"mean = {mean:.1f} frames")
    plt.xlabel("Length (frames)")
    plt.ylabel("Count")
    plt.title(title + f"  (N={n})")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=160)
    if show:
        plt.show()
    return save_path



# -----------------------------------------------------------------------------#
# Dimensionality reduction
# -----------------------------------------------------------------------------#

def reduce_dim(X: np.ndarray,
               cfg: Config) -> Tuple[np.ndarray, PCA]:
    """PCA to desired retained variance."""
    pca = PCA(n_components=cfg.pca_variance,
              random_state=cfg.random_state,
              svd_solver='full')
    Xr = pca.fit_transform(X)
    return Xr, pca


# -----------------------------------------------------------------------------#
# Clustering wrappers
# -----------------------------------------------------------------------------#

def cluster_kmeans(X: np.ndarray,
                   k: int,
                   cfg: Config) -> np.ndarray:
    model = KMeans(n_clusters=k,
                   n_init="auto",
                   random_state=cfg.random_state)
    return model.fit_predict(X)


def cluster_spectral(X: np.ndarray,
                     k: int,
                     n_neighbors: int,
                     cfg: Config) -> np.ndarray:
    model = SpectralClustering(
        n_clusters=k,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=cfg.random_state,
    )
    return model.fit_predict(X)

def cluster_dbscan(X_red: np.ndarray,
                   cfg,
                   eps: float | None = None,
                   min_samples: int | None = None):
    """
    Density-based clustering (DBSCAN) that matches the data contract of
    cluster_kmeans() / cluster_spectral().

    Parameters
    ----------
    X_red : ndarray            # already reduced feature space
    cfg   : Namespace / dict   # same cfg object you pass elsewhere
    eps, min_samples :         # override cfg defaults if supplied
    Returns
    -------
    labels : ndarray, shape (n_segments,)
    model  : fitted DBSCAN instance
    """
    eps_ = eps or getattr(cfg, "dbscan_eps", 0.5)
    ms_  = min_samples or getattr(cfg, "dbscan_min_samples", 5)

    model = DBSCAN(eps=eps_, min_samples=ms_, metric="cosine", n_jobs=-1)
    labels = model.fit_predict(X_red)

    # keep the return tuple identical to k-means
    return labels

def estimate_eps(X, k=5):
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    kth = np.sort(d[:, -1])          # distance to k-th neighbour
    return np.percentile(kth, 75)    # robust elbow


def tune_dbscan(X, cfg):
    eps0 = estimate_eps(X, cfg.dbscan_min_samples)
    grid = eps0 * np.array([0.8, 1.0, 1.2, 1.4])
    best_eps, best_labels, best_k = None, None, -1
    for eps in grid:
        labels = cluster_dbscan(X, cfg, eps=eps)
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k > best_k:
            best_eps, best_k, best_labels = eps, k, labels
    print(f"[DBSCAN] eps*={best_eps:.2f} → {best_k} clusters (+noise)")
    return best_eps, best_labels


def cluster_hierarchical_ward(X_red: np.ndarray,
                              cfg,
                              n_clusters: int | None = None):
    """
    Ward-linkage agglomerative clustering. Mirrors the k-means interface.
    """
    k_ = n_clusters or getattr(cfg, "hier_n_clusters", 5)

    model = AgglomerativeClustering(
        n_clusters=k_,
        metric="euclidean",
        linkage="ward")
    labels = model.fit_predict(X_red)

    return labels

def cluster_two_step_hierarchical(X_red: np.ndarray,
                                  cfg,
                                  first_k: int | None = None,
                                  final_k: int | None = None):
    """
    Two-stage Ward: (1) over-cluster, (2) cluster the cluster centroids.
    """
    first_k  = first_k  or getattr(cfg, "first_k", 20)
    final_k  = final_k  or getattr(cfg, "final_k", 5)

    # ---------- Stage 1 ----------
    stage1 = AgglomerativeClustering(
        n_clusters=first_k, linkage="ward", metric="euclidean"
    ).fit_predict(X_red)

    # Compute centroids of those micro-clusters
    centroids = np.vstack([
        X_red[stage1 == cid].mean(axis=0) for cid in range(first_k)
    ])

    # ---------- Stage 2 ----------
    stage2_model = AgglomerativeClustering(
        n_clusters=final_k, linkage="ward", metric="euclidean"
    )
    stage2_labels = stage2_model.fit_predict(centroids)

    # broadcast centroid labels back to individual segments
    labels = np.array([stage2_labels[cid] for cid in stage1])

    return labels

def ward_by_distance(X_red: np.ndarray, dist: float) -> np.ndarray:
    """SciPy linkage + fcluster → labels at a given cut-height."""
    Z = linkage(X_red, method="ward", metric="euclidean")
    labels = fcluster(Z, t=dist, criterion="distance") - 1  # 0-based
    return labels

def total_variance(X, labels):
    var = 0.0
    for lab in set(labels):
        var += np.var(X[labels == lab])
    return var

def tune_ward_distance(X_red, cfg):
    best_cost, best_labels, best_d = np.inf, None, None
    for d in cfg.hier_grid:
        labels = ward_by_distance(X_red, d)
        k = len(set(labels))
        cost = total_variance(X_red, labels) + cfg.hier_penalty * k
        if cost < best_cost:
            best_cost, best_labels, best_d = cost, labels, d
    print(f"[Ward] dist*={best_d} → {k} clusters, cost={best_cost:.2f}")
    return best_d, best_labels

def tune_two_step(X_red, cfg):
    best_cost, best_labels = np.inf, None
    for broad in cfg.hier_grid:
        coarse = ward_by_distance(X_red, broad)              # 1️⃣ coarse cut
        cents  = np.vstack([X_red[coarse == c].mean(axis=0)  # centroids
                            for c in sorted(set(coarse))])
        for refine in (broad/4, broad/3, broad/2):           # 2️⃣ refine grid
            fine = ward_by_distance(cents, refine)
            labels = np.array([fine[c] for c in coarse])
            cost = total_variance(X_red, labels) + cfg.hier_penalty * len(set(labels))
            if cost < best_cost:
                best_cost, best_labels = cost, labels
    print(f"[2-step Ward] cost={best_cost:.2f} → {len(set(best_labels))} clusters")
    return best_labels




# -----------------------------------------------------------------------------#
# Export helpers
# -----------------------------------------------------------------------------#



def write_manifest(segs: List[Tuple[int, int]],
                   labels: np.ndarray,
                   path: Path):
    """
    CSV with start_sample, end_sample, label.
    """
    header = "start_sample,end_sample,label\n"
    rows = [f"{s},{e},{lab}\n" for (s, e), lab in zip(segs, labels)]
    path.write_text(header + "".join(rows), encoding='utf-8')
    

def save_cluster_snippets(audio_path: Path,
                          sr: int,
                          segs: List[Tuple[int, int]],
                          labels: np.ndarray,
                          out_dir: Path,
                          concat: bool = True):
    """
    Write individual WAVs (cluster_<id>/<start>_<end>.wav).  If `concat=True`,
    also stream all snippets of a cluster into cluster_<id>.wav.
    Guaranteed to emit **mono** data regardless of the source file’s channel
    count, so the concat handle is always opened with channels=1.
    """
    concat_handles: dict[int, sf.SoundFile] = {}

    with sf.SoundFile(audio_path) as snd, \
         tqdm(total=len(segs), desc=f"Snippets → {out_dir.name}", unit="wav") as bar:

        for (start, end), lab in zip(segs, labels):
            snd.seek(start)
            # read strictly as 2-D (frames, channels)
            clip = snd.read(end - start, dtype='float32', always_2d=True)
            # fold to mono and ensure C-contiguous 1-D shape
            clip = np.asarray(clip.mean(axis=1), order='C')

            # ---------- disk I/O ----------
            if concat:
                wav_path = out_dir / (f"cluster_{lab:03d}.wav" if lab >= 0 else "cluster_noise.wav")
                if lab not in concat_handles:
                    concat_handles[lab] = sf.SoundFile(
                        wav_path, mode='w',
                        samplerate=sr, channels=1, subtype='PCM_16')
                concat_handles[lab].write(clip)

            bar.update(1)

    for h in concat_handles.values():
        h.close()

# -----------------------------------------------------------------------------
# Visualization helpers: overview & per-call spectrograms
# -----------------------------------------------------------------------------

def plot_db_spectrogram(
    y: np.ndarray,
    sr: int,
    cfg,
    title: str = "",
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    show: bool = True,
    save_path: Optional[Path] = None,
):
    """
    Band-pass (cfg.bandpass_hz) -> STFT -> log-dB spectrogram.
    Uses cfg.n_fft / cfg.hop_fft.
    """
    y_f = bandpass(y, sr, *cfg.bandpass_hz)
    f, t, Z = stft(
        y_f,
        fs=sr,
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_fft,
        window="hann",
        padded=False,
        boundary=None,
    )
    S = 10.0 * np.log10((np.abs(Z) ** 2) + cfg.ref_db_floor)

    if vmax is None:
        vmax = np.percentile(S, 99.0)
    if vmin is None:
        vmin = vmax - 80.0  # ~80 dB range looks good for bird calls

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f / 1000.0, S, shading="auto", vmin=vmin, vmax=vmax)
    plt.ylim(cfg.bandpass_hz[0] / 1000.0, cfg.bandpass_hz[1] / 1000.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Power (dB)")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def save_overview_spectrogram(
    audio_path: Path,
    sr: int,
    segs: List[Tuple[int, int]],
    cfg,
    out_dir: Path,
    preview_sec: int = 15,
) -> Tuple[Path, Path]:
    """
    Save (1) a band-passed log-dB spectrogram over a short window and
    (2) the matching waveform with vertical lines at detected segment bounds.
    Returns (spectrogram_png, waveform_png).
    """
    assert len(segs) > 0, "No segments to preview."

    first_start = segs[0][0]
    win_start = max(0, first_start - (preview_sec // 2) * sr)
    win_end = win_start + preview_sec * sr

    with sf.SoundFile(audio_path) as snd:
        snd.seek(win_start)
        y_win = snd.read(win_end - win_start, dtype="float32", always_2d=True).mean(axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "overview.png"
    plot_db_spectrogram(
        y_win,
        sr,
        cfg,
        title=f"Overview spectrogram (t ≈ {win_start/sr:.1f}–{win_end/sr:.1f}s)",
        show=False,
        save_path=spec_path,
    )

    wave_path = out_dir / "overview_wave_with_segments.png"
    t_axis = np.arange(len(y_win)) / sr
    plt.figure(figsize=(10, 1.4))
    plt.plot(t_axis, y_win, linewidth=0.5)
    for (s, e) in segs:
        if s >= win_start and e <= win_end:
            t0 = (s - win_start) / sr
            t1 = (e - win_start) / sr
            plt.axvline(t0, linestyle="--", linewidth=0.8)
            plt.axvline(t1, linestyle="--", linewidth=0.8)
    plt.xlabel("Time (s)   (same window as the spectrogram)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(wave_path, dpi=150)
    plt.close()

    return spec_path, wave_path


def save_call_spectrograms(
    audio_path: Path,
    sr: int,
    segs: List[Tuple[int, int]],
    cfg,
    out_dir: Path,
    labels: Optional[np.ndarray] = None,
    n_preview: int = 20,
    mode: str = "random",  # "random" or "longest"
) -> List[Path]:
    """
    Save band-passed log-dB spectrogram PNGs for a subset of detected calls.
    If `labels` provided, include them in the filename & title.
    """
    if len(segs) == 0:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    seg_lengths = np.array([e - s for (s, e) in segs])

    if mode == "longest":
        idx = np.argsort(seg_lengths)[-min(n_preview, len(segs)) :]
    else:
        rng = np.random.default_rng(getattr(cfg, "random_state", None))
        n_take = min(n_preview, len(segs))
        idx = rng.choice(len(segs), size=n_take, replace=False)
        idx.sort()

    saved: List[Path] = []
    with sf.SoundFile(audio_path) as snd:
        for i in idx:
            s, e = segs[i]
            snd.seek(s)
            y_seg = snd.read(e - s, dtype="float32", always_2d=True).mean(axis=1)

            lab_txt = ""
            if labels is not None:
                lab_txt = f"_km{int(labels[i])}"
            title = f"Call #{i}  ({(e - s) / sr * 1000:.0f} ms){' | k-means: ' + str(int(labels[i])) if labels is not None else ''}"

            out_png = out_dir / f"call_{i:05d}{lab_txt}.png"
            plot_db_spectrogram(y_seg, sr, cfg, title=title, show=False, save_path=out_png)
            saved.append(out_png)

    return saved

# -----------------------------------------------------------------------------#
# Duration analytics & human-validation audio
# -----------------------------------------------------------------------------#

from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

def compute_segment_durations(
    segs: List[Tuple[int, int]],
    sr: int,
    unit: str = "ms",
) -> np.ndarray:
    """
    Return a 1-D array of segment durations given global sample-indexed segments.

    Parameters
    ----------
    segs : list of (start_sample, end_sample)
        Segment bounds in GLOBAL samples (as produced by `detect_segments`).
    sr : int
        Sampling rate (Hz).
    unit : {'s','ms'}
        Output unit for durations (seconds or milliseconds).

    Returns
    -------
    durations : (N,) float ndarray
        Durations in requested unit.
    """
    if len(segs) == 0:
        return np.array([], dtype=np.float32)
    dur_s = (np.array([e - s for (s, e) in segs], dtype=np.float64) / float(sr))
    if unit == "s":
        return dur_s.astype(np.float32)
    elif unit == "ms":
        return (dur_s * 1_000.0).astype(np.float32)
    else:
        raise ValueError("unit must be 's' or 'ms'")


def plot_duration_histogram(
    durations: np.ndarray,
    bins: int | str = "auto",
    unit: str = "ms",
    log_x: bool = False,
    show: bool = True,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Plot a histogram of call durations with median/mean markers.

    Parameters
    ----------
    durations : (N,) ndarray
        Durations in the desired unit (use `compute_segment_durations` first).
    bins : int or 'auto'
        Histogram binning.
    unit : {'s','ms'}
        Labeling only; does not convert values.
    log_x : bool
        If True, sets x-axis to log-scale (helpful for heavy tails).
    show : bool
        If True, display the plot.
    save_path : Optional[Path]
        If provided, saves the figure here (directories auto-created).
    title : Optional[str]
        Custom title; defaults to "Call duration distribution (N=...)".
    """
    N = len(durations)
    if N == 0:
        raise ValueError("No durations to plot.")

    med = float(np.median(durations))
    mean = float(np.mean(durations))

    plt.figure(figsize=(8, 3))
    plt.hist(durations, bins=bins, edgecolor="none")
    plt.axvline(med, linestyle="--", linewidth=1.2, label=f"median = {med:.1f} {unit}")
    plt.axvline(mean, linestyle=":", linewidth=1.2, label=f"mean = {mean:.1f} {unit}")
    if log_x:
        plt.xscale("log")
    plt.xlabel(f"Duration ({unit})")
    plt.ylabel("Count")
    plt.title(title or f"Call duration distribution (N={N})")
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    return save_path


def concat_calls_sorted_by_duration(
    audio_path: Path,
    sr: int,
    segs: List[Tuple[int, int]],
    out_wav: Path,
    gap_ms: int = 100,
    limit: Optional[int] = None,
) -> Path:
    """
    Write a single mono WAV that concatenates ALL detected calls in ascending
    duration order, inserting a fixed silence gap between calls.

    This is intended for quick human validation of the segmenter.

    Parameters
    ----------
    audio_path : Path
        Path to the source audio (24 h file).
    sr : int
        Sampling rate in Hz. Must match the file's sample rate.
    segs : list[(start_sample, end_sample)]
        Segment bounds in GLOBAL sample units.
    out_wav : Path
        Destination WAV path. Will be created/overwritten.
    gap_ms : int
        Silence inserted between consecutive calls.
    limit : Optional[int]
        If set, only the shortest `limit` calls are included (for quick preview).

    Returns
    -------
    out_wav : Path
        The path to the written WAV file.
    """
    if len(segs) == 0:
        raise ValueError("No segments provided.")

    # Sort by duration ascending
    durations = np.array([e - s for (s, e) in segs], dtype=np.int64)
    order = np.argsort(durations, kind="stable")
    if limit is not None:
        order = order[: int(limit)]

    gap_len = int(round(gap_ms * 1e-3 * sr))
    gap = np.zeros(gap_len, dtype=np.float32) if gap_len > 0 else None

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Stream read/write to keep memory bounded; fold to mono (match your pattern).
    # (You already do this in save_cluster_snippets.) 【turn4file11†library.py†L38-L55】
    with sf.SoundFile(audio_path) as snd, \
         sf.SoundFile(out_wav, mode="w", samplerate=sr, channels=1, subtype="PCM_16") as out, \
         tqdm(total=len(order), desc=f"Concat→{out_wav.name}", unit="call") as bar:

        # Basic format sanity
        if snd.samplerate != sr:
            raise ValueError(f"Expected {sr} Hz but file has {snd.samplerate} Hz")

        for idx in order:
            s, e = segs[int(idx)]
            if e <= s:
                bar.update(1)
                continue

            snd.seek(s)
            clip = snd.read(e - s, dtype="float32", always_2d=True).mean(axis=1)
            out.write(np.asarray(clip, order="C"))

            if gap is not None and len(gap) > 0:
                out.write(gap)

            bar.update(1)

    return out_wav
