import soundfile as sf
from preprocessing import segment_bird_calls, produce_spectrogram, apply_band_pass_filter
import numpy as np
import librosa
from tqdm import tqdm
from constants import *
import os


def mfcc_vector(y, sr, n_mfcc=20):
    """Return 40-D vector [µ, σ] of MFCCs for one call snippet."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])

#for large files, we read them in chunks to avoid memory issues
def segment_large_file(path, sr, *, chunk_sec=600, baseline_lin=None, **seg_kwargs):
    segments = []
    offset_frames = 0

    with sf.SoundFile(path) as f:
        chunk_frames = int(chunk_sec * sr)
        total_frames = f.frames
        pbar = tqdm(total=total_frames,
                   unit="sample",
                   unit_scale=True,
                   desc="Detecting segments")
        
        while True:
            y = f.read(frames=chunk_frames, dtype="float32", always_2d=True)
            if y.size == 0:
                break                                    # EOF
            if y.ndim == 2:
                y = ((y[:, 0] + y[:, 1]) * 0.5).astype(np.float32)

            # run existing detector on this slice
            segs = segment_bird_calls(
                y, sr,
                baseline_lin=baseline_lin,
                amp_range_db=AMP_RANGE_DB,
                **seg_kwargs
            )
            # shift every time stamp to absolute timeline
            for s in segs:
                shift = offset_frames / sr
                s["start_s"] += shift
                s["end_s"] += shift
                s["peak_s"] += shift
            segments.extend(segs)

            offset_frames += y.shape[0]
            pbar.update(y.shape[0])
    pbar.close()
    return segments

def build_feature_matrix_streaming(path, segments, sr=48_000):
    """
    Read each detected call straight from disk, convert to a fixed-length
    log-power patch (+ stats), stack into a 2-D array.
    """
    X = []
    with sf.SoundFile(path) as f:
        for seg in tqdm(segments, desc="Building feature matrix", unit="call"):
            f.seek(int(seg["start_s"] * sr))
            frames = int((seg["end_s"] - seg["start_s"]) * sr)
            y = f.read(frames, dtype="float32", always_2d=True)
            if y.ndim == 2:
                y = ((y[:, 0] + y[:, 1]) * 0.5).astype(np.float32)
            X.append(spec_mfcc_vector(y, sr))
    return np.vstack(X).astype(np.float32)

def pad_signals_to_uniform_length(signals, pad_value=0):
    """Zero-pad each 1D array in `signals` to the same length."""
    max_len = max(len(sig) for sig in signals)
    padded = []
    for sig in signals:
        pad = max_len - len(sig)
        padded.append(np.pad(sig, (0, pad), constant_values=pad_value))
    return np.vstack(padded)

def pad_spectrograms_to_uniform_time(specs, pad_value=0):
    """
    specs: list[ndarray]  each of shape (freq × time)
    Returns a 3-D array with equalised time dimension.
    """
    max_T = max(s.shape[1] for s in specs)
    padded = []
    for S in specs:
        pad = max_T - S.shape[1]
        padded.append(np.pad(S, ((0,0), (0,pad)),
                           constant_values=pad_value))
    return np.stack(padded)

def spec_patch_vector(y, sr, *, time_bins: int = TIME_BINS_PATCH,
                     rms_normalise: bool = False, whiten: bool = True):
    """
    Returns 1-D vector = flattened log-power spectrogram (+3 summary stats)
    Stats appended: rms (dB), spectral centroid, spectral bandwidth
    """
    # 1 Band-pass
    y_bp = apply_band_pass_filter(y, sr, F_BAND[0], F_BAND[1])

    # 2 Optional RMS normalisation  (turn off by default)
    if rms_normalise:
        y_bp = y_bp / (np.sqrt(np.mean(y_bp**2)) + 1e-12)

    # 3 Spectrogram in **linear power**
    S_pow, freqs, times = produce_spectrogram(
        y_bp, sr,
        scale_out="power",
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # 4 Pad / trim exactly like reference.ipynb
    #    (librosa.util.fix_length handles both cases)
    # 4 Pad / trim (constant zero adds −inf dB after conversion)
    S_pow = librosa.util.fix_length(
        S_pow, size=time_bins, axis=1, constant_values=0.0
    )


    # 5 Convert to dB *after* all linear maths
    S_db = librosa.power_to_db(S_pow, ref=1.0)
    if whiten:
    # If whiten is True, each frequency row of the dB patch is de-meaned and
    # divided by its own standard deviation before flattening.
        S_db -= S_db.mean(axis=1, keepdims=True)
        S_db /= (S_db.std(axis=1, keepdims=True) + 1e-12)

    # 6 Shape statistics (computed in linear domain → dB last)
    cent = librosa.feature.spectral_centroid(S=S_pow, sr=sr).mean()
    bw = librosa.feature.spectral_bandwidth(S=S_pow, sr=sr).mean()
    rms_lin = np.mean(y_bp**2)
    rms_db = librosa.power_to_db(np.array([rms_lin]), ref=1.0)[0]

    # 7 Return flattened vector + stats
    flat = S_db.flatten()
    return np.hstack([flat, rms_db, cent, bw])

# ── NEW: spectrogram + MFCC hybrid ─────────────────────────────────────────
def spec_mfcc_vector(
        y, sr,
        *,
        time_bins: int = TIME_BINS_PATCH,
        rms_normalise: bool = False,
        whiten: bool = True,
        n_mfcc: int = 20
):
    """
    Concatenate the existing spec-patch feature with a 40-D MFCC [µ, σ] vector.
    Returns 1-D ndarray of length (freq·time)+3 + 2·n_mfcc.
    """
    # 1) Existing log-power patch (+3 summary stats)
    spec_feat = spec_patch_vector(
        y, sr,
        time_bins=time_bins,
        rms_normalise=rms_normalise,
        whiten=whiten
    )

    # 2) MFCC statistics (uses raw *band-passed* audio for robustness)
    y_bp = apply_band_pass_filter(y, sr, F_BAND[0], F_BAND[1])
    mfcc_feat = mfcc_vector(y_bp, sr, n_mfcc=n_mfcc)

    # 3) Concatenate
    return np.hstack([spec_feat, mfcc_feat])



def export_cluster_audio(
        audio_path        : str | os.PathLike,
        segments          : list[dict],
        labels            : np.ndarray,
        cluster_id        : int,
        out_wav           : str | os.PathLike,
        *,
        sr                : int        = 48_000,
        max_calls         : int | None = 100,   # keep files short
        pad_sec           : float      = 0.05   # silence between calls
) -> None:
    """Write one WAV containing (up to) *max_calls* snippets of a cluster.

    Calls are separated by *pad_sec* seconds of zero.
    """
    import soundfile as sf

    pad = np.zeros(int(pad_sec * sr), dtype=np.float32)
    out_chunks = []

    # choose only calls with the requested label
    idxs = np.where(labels == cluster_id)[0]
    if max_calls is not None:
        idxs = idxs[:max_calls]

    with sf.SoundFile(audio_path) as f:
        for i in idxs:
            seg = segments[i]
            start_fr = int(seg["start_s"] * sr)
            dur_fr   = int((seg["end_s"] - seg["start_s"]) * sr)

            f.seek(start_fr)
            y = f.read(dur_fr, dtype="float32", always_2d=True)
            if y.ndim == 2:
                y = y.mean(axis=1)

            out_chunks.extend([y, pad])

    if not out_chunks:
        print(f"[cluster {cluster_id}] — zero snippets, nothing written")
        return

    concat = np.concatenate(out_chunks)
    sf.write(out_wav, concat, samplerate=sr)
    print(f"✔ saved {out_wav}  ({len(idxs)} calls, {concat.shape[0]/sr:.1f} s)")

