# %%
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import librosa.display
from preprocessing import segment_bird_calls, produce_spectrogram
import numpy as np
from pathlib import Path
from analysis import build_feature_matrix, pipeline_pca_kmeans, plot_pca_clusters, pipeline_pca_spectral
from helper import segment_large_file, build_feature_matrix_streaming, export_cluster_audio
import collections

# %%
# ── CONFIG ────────────────────────────────────────────────────────────
# turn the relative string into an absolute path *once*
AUDIO_FILE = (Path(__file__).parent / ".." / "data" / "nestingbox1" /
              "20240927_204919" / "combined_audio.w64").resolve(strict=True)
SAMPLE_RATE  = 48_000                 # Hz   (keep one single source of truth)
BANDPASS_HZ  = (2_000, 8_000)
CHUNK_SEC    = 7200                    # length fed to segment_large_file
METHOD       = "rms"                  
K_RANGE_PCA  = range(3, 10)
BASE_OUT = Path(__file__).parent / "cluster_wavs"      # master directory
KM_OUT   = BASE_OUT / "kmeans"                           # k-means sub-dir
SP_OUT   = BASE_OUT / "spectral"                         # spectral sub-dir
for d in (KM_OUT, SP_OUT):
    d.mkdir(parents=True, exist_ok=True)
THRESH_MULT       = 4.0          # energy must exceed baseline × 4


# %% ─────────────────────── 3. Helper wrappers ─────────────────────
def load_segments() -> list[dict]:
    """Cut a very long file into candidate calls (RMS detector)."""
    return segment_large_file(
        AUDIO_FILE,
        sr=SAMPLE_RATE,
        chunk_sec=CHUNK_SEC,
        band=BANDPASS_HZ,
        method=METHOD,
        baseline_lin=None,
    )



print("✅ Helper utilities ready")

# %% Load segments

segments = load_segments()
print(f"Detected {len(segments)} segments")
# Build feature matrix (streaming → constant RAM)
X = build_feature_matrix_streaming(AUDIO_FILE, segments, sr=SAMPLE_RATE)


# %% PCA&Kmeans

pca_res = pipeline_pca_kmeans(X, k_range=K_RANGE_PCA)
labels  = pca_res["labels"]
X_red   = pca_res["pca"].transform(pca_res["scaler"].transform(X))

print(f"✅ Chose k = {pca_res['best_k']} (PCA / k-means)")

# Visualise
plot_pca_clusters(
    X_red,
    labels,
    pca_res["pca"].explained_variance_ratio_,
    dims=(0, 1),
)
# export audio for each cluster
unique_labels = sorted(set(labels))
for cid in unique_labels:
    export_cluster_audio(
        AUDIO_FILE,
        segments,
        labels,
        cluster_id = cid,
        out_wav    = KM_OUT / f"cluster_{cid:02d}.wav",
        sr         = SAMPLE_RATE,
        max_calls  = 100          # tweak or set None for all
    )

# %% Spectral Clustering (graph-based Laplacian)
spec_res  = pipeline_pca_spectral(X, k_range=K_RANGE_PCA)
spec_k    = spec_res["best_k"]
spec_lab  = spec_res["labels"]
spec_Xred = spec_res["X_red"]

print(f"✅ Spectral (graph) clustering chose k = {spec_k}")

# Visualise on the same PC axes
plot_pca_clusters(
    spec_Xred,
    spec_lab,
    spec_res["pca"].explained_variance_ratio_,
    dims=(0, 1),
)

unique_labels = sorted(set(spec_lab))
for cid in unique_labels:
    out_path = SP_OUT / f"cluster_{cid:02d}.wav"
    export_cluster_audio(
        AUDIO_FILE,
        segments,
        spec_lab,
        cluster_id = cid,
        out_wav    = out_path,
        sr         = SAMPLE_RATE,
        max_calls  = 100          # tweak or set None for all
    )




# %% ───────────────── 7. Optional: full-file spectrogram ───────────
# Uncomment to visualize the spectrogram of the selected window of the audio file:
START_SEC = 0      # 0 seconds in
WINDOW_SEC = 120          # 5 minutes window
with sf.SoundFile(AUDIO_FILE) as f:
    sr = f.samplerate
    f.seek(int(START_SEC * sr))
    y = f.read(int(WINDOW_SEC * sr), dtype="float32", always_2d=False)
if y.ndim == 2:
    y = y.mean(axis=1)
S_db, freqs, times = produce_spectrogram(
    y, sr,
    method="stft",
    scale_out="db",
)
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, x_axis="time",
                         y_axis="log", cmap="magma")
plt.colorbar(label="dB")
plt.title("Full-file spectrogram (log-freq)")
plt.tight_layout(); plt.show()

# %%
