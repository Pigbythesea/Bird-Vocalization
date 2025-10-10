import librosa
import soundfile as sf
import numpy as np
import scipy.signal as signal
import pywt 
from constants import N_FFT, HOP_LENGTH, F_BAND
from tqdm import tqdm
from constants import WIN_LENGTH, HOP_LENGTH, THRESH_MULT, PATCH_DUR_SEC, F_BAND



AMP_RANGE_DB  = (-80, -1)   # keep peaks within this dB window
OVERLAP_SEC   = 1.0          # overlap between 2-h chunks



def apply_band_pass_filter(data, sr, lowcut, highcut, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def produce_spectrogram(
    data: np.ndarray,
    sr: int,
    *,
    method: str = "stft",
    n_fft: int = 2048,
    hop_length: int = 512,
    wavelet: str = "cmor",
    scales: np.ndarray | None = None,
    max_duration: float | None = None,
    scale_out: str = "db",           # "db", "power", "amplitude"
):
    """
    Compute a magnitude spectrogram (STFT) or scalogram (CWT).

    Parameters
    ----------
    y : ndarray
        Audio signal already loaded in *main.py* (e.g. via `librosa.load`).
    sr : int
        Its sampling rate.
    method : {"stft", "wavelet"}, default="stft"
        Transform algorithm.
    n_fft, hop_length : int
        STFT parameters (ignored for CWT).
    wavelet : str, default="cmor"
        Mother wavelet for CWT.
    scales : ndarray | None
        Custom scale vector for CWT; defaults to ``np.arange(1, 64)``.
    max_duration : float | None
        If set, truncate *y* to that many seconds first (handy for
        quick sanity checks on a 24-h clip).
    scale_out : {"db", "power", "amplitude"}
        Magnitude scaling for STFT output.

    Returns
    -------
    S : ndarray
        Transform magnitude (freq × time).
    freqs : ndarray
        Frequency axis in Hz (STFT) or pseudo-freq (CWT).
    times : ndarray
        Time axis in seconds.
    """
    if max_duration is not None:
        data = data[: int(max_duration * sr)]

    if method.lower() == "stft":
        D      = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
        mag    = np.abs(D)                       # √(real²+imag²)
        power  = mag ** 2                        # linear Power
        if   scale_out == "power":
            S = power
        elif scale_out == "db":
            S = librosa.power_to_db(power, ref=np.max)
        else:                                    # "amplitude"
            S = mag
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(
                    np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
        return S, freqs, times

    # ── Continuous-wavelet option (rarely used here) ──
    if method.lower() == "wavelet":
        scales = np.arange(1, 64) if scales is None else scales
        coef, freqs = pywt.cwt(data, scales, wavelet,
                               sampling_period=1 / sr)
        S      = np.abs(coef)
        times  = np.linspace(0, len(data) / sr, S.shape[1],
                             endpoint=False)
        return S, freqs, times

    raise ValueError(f"Unsupported method: {method!r}")

def segment_bird_calls(
    data: np.ndarray,
    sr: int,
    *,
    baseline_lin: float,                      
    band: tuple[float, float] = (2_000, 8_000),
    method: str = "rms",
    frame_length: int = 2_048,
    hop_length: int = 512,
    peak_distance: float = 0.15,
    amp_range_db: tuple[float, float] | None = None,
    min_call_dur: float = 0.05,
    merge_gap: float = 0.20
) -> list[dict]:
    """
    Detect and segment putative bird calls.

    Returns
    -------
    segments : list of dict
        Each dict has keys {"start_s", "end_s", "peak_s", "peak_level_db"}.
    """
    # 1) Band-pass filter the audio
    y_bp = apply_band_pass_filter(
        data, sr, band[0], band[1]
    ) 
    # 2) Energy (1-D) over time
    hop = hop_length
    if method == "rms":
        rms         = librosa.feature.rms(y=y_bp,
                                      frame_length=frame_length,
                                      hop_length=hop_length,
                                      center=True).squeeze()
        energy_lin  = rms ** 2
    elif method == "stft":
        D           = librosa.stft(y_bp, n_fft=frame_length,
                               hop_length=hop_length, center=True)
        mag         = np.abs(D)
        power       = mag ** 2
        energy_lin  = np.sum(power, axis=0)          # linear Power
    else:
        raise ValueError("method must be 'rms' or 'stft'")
    
    # ─ local baseline, linear units ─
    baseline_lin = float(np.median(energy_lin)) if baseline_lin is None else baseline_lin
    peak_height_lin = baseline_lin * THRESH_MULT
    amp_thresh_lin  = baseline_lin * (THRESH_MULT / 2)   # hysteresis


    distance_frames = int(np.round(peak_distance * sr / hop))

    peaks, _ = signal.find_peaks(
    energy_lin,
    height=peak_height_lin,
    distance=max(1, distance_frames)
    )
    
    if amp_range_db is not None:
        low, high = amp_range_db
        peaks_kept = []
        for p in peaks:
            lev_db = librosa.power_to_db([energy_lin[p]], ref=np.max)[0]
            if low <= lev_db <= high:
                peaks_kept.append(p)
        peaks = np.array(peaks_kept, dtype=int)

    # 5) Grow each peak left/right until we cross the baseline threshold
    to_sec = lambda idx: idx * hop / sr
    raw_segments = []
    for p in peaks:
        left = p
        while left > 0 and energy_lin[left] >= amp_thresh_lin:
            left -= 1
        right = p
        while right < len(energy_lin) - 1 and energy_lin[right] >= amp_thresh_lin:
            right += 1

        if (right - left) * hop / sr >= min_call_dur:
            raw_segments.append(
                dict(
                    start_s=to_sec(left),
                    end_s=to_sec(right),
                    peak_s=to_sec(p),
                    peak_level_db=float(librosa.power_to_db(
                          np.array([energy_lin[p]]), ref=np.max)),
                )
            )

    # 6) Merge neighbouring segments with tiny gaps
    segments: list[dict] = []
    for seg in sorted(raw_segments, key=lambda d: d["start_s"]):
        if not segments:
            segments.append(seg)
            continue
        prev = segments[-1]
        if seg["start_s"] - prev["end_s"] <= merge_gap:
            # extend previous
            prev["end_s"] = seg["end_s"]
            if seg["peak_level_db"] > prev["peak_level_db"]:
                prev["peak_s"] = seg["peak_s"]
                prev["peak_level_db"] = seg["peak_level_db"]
        else:
            segments.append(seg)

    return segments

def _detect_peaks_stft_dyn(y, sr, *, n_fft=N_FFT, hop=HOP_LENGTH,
                           hpf_cutoff=900, sigma_thresh=3):
    """
    High-pass > hpf_cutoff, STFT-sum energy trace, local-baseline σ rule.
    Returns list of (start_idx, end_idx, peak_idx, peak_linpow).
    """
    y_hp = apply_high_pass_filter(y, sr, cutoff=hpf_cutoff)   # reuse your filt util
    S_lin, *_ = produce_spectrogram(y_hp, sr,
                                    n_fft=n_fft,
                                    hop_length=hop,
                                    scale_out="power")
    energy = S_lin.sum(axis=0)
    # local baseline: midpoint between successive peaks
    baseline = scipy.ndimage.uniform_filter1d(energy, size=301)
    resid    = energy - baseline
    sigma    = np.std(resid)
    thresh   = baseline + sigma_thresh * sigma
    peaks, _ = scipy.signal.find_peaks(energy, height=thresh)
    # convert peak frames to sample indices and form segments …
