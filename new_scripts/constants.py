# ---------- spectrogram parameters ----------
SAMPLE_RATE      = 48_000          
N_FFT            = 1_024        # FFT window size
WIN_LENGTH       = 1_024        # FFT window size
HOP_LENGTH       = 512         
TIME_BINS_PATCH  = 128             # all snippets padded / trimmed to this
F_BAND           = (2_000, 8_000) 
AMP_RANGE_DB     = (-80, -0)   # keep peaks within this dB window
# ---------- detector / patch parameters ----------
THRESH_MULT       = 4.0          # energy must exceed baseline × 4
PATCH_DUR_SEC     = 0.6          # fixed snippet duration
RMS_NORMALISE     = False        # turn on for ablation tests
N_MFCC            = 20
