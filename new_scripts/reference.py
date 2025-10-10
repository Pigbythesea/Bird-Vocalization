# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.signal
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from pydub import AudioSegment
import pywt
import optimalK
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score
from collections import Counter
import itertools
import os
import pickle
import random

# %%
# deprecated
def get_high_pass_filter(cutoff_freq, sr):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    return scipy.signal.butter(1, normal_cutoff, btype='high', analog=False)

# a band-stop filter currently set as a high-pass filter
def band_stop_filter(y, sr, highcut, order=4):
    nyquist = 0.5 * sr
    normal_cutoff = highcut / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    y_filtered = scipy.signal.filtfilt(b, a, y)
    return y_filtered

# %%
# target audio file
file_path = '../audios/Barn Owl Nestling\'s Adorable Flight Training Session!.mp3'
y, sr = librosa.load(file_path)

# y = band_stop_filter(y, sr, highcut=900)

# %% [markdown]
# 

# %%
def produce_spectrogram(y_filtered, sr, cutoff_freq, method='stft', dur=20, vmin=0, vmax=1):
    if dur != -1:
        y_filtered = y_filtered[:int(dur * sr)]
    if method == 'stft':
        D_filtered = librosa.stft(y_filtered)
        S_raw_filtered = np.abs(D_filtered)
#         S_raw_filtered *= 10

#         print(f"Shape of D_filtered: {D_filtered.shape}")
#         print(f"Shape of S_raw_filtered: {S_raw_filtered.shape}")

        plt.figure(figsize=(12, 8))
        
        # Plot the spectrogram
        img = librosa.display.specshow(S_raw_filtered, sr=sr, x_axis='time', y_axis='log')

        plt.ylim(512, sr // 2)

        # Set the color limits for better visibility
        plt.clim(vmin=vmin, vmax=vmax)

        # Add color bar and customize the tick formatting to reflect amplitude values
        cbar = plt.colorbar(img, format='%+1.2f')  # Only one colorbar here

        # Add labels and title
        plt.title(f'Spectrogram (Frequency vs. Time) with High-Pass Filter at {cutoff_freq}Hz')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()
        
    # similar ideas for wavelet transform
    if method == 'wavelet':
        scales = np.arange(1, 64)
        
        coef, freq = pywt.cwt(y_filtered, scales, 'cmor', sampling_period=1/sr)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(np.abs(coef), extent=[0, dur, scales[-1], scales[0]], cmap='viridis', aspect='auto',
                   vmax=np.abs(coef).max(), vmin=-np.abs(coef).max())
        plt.colorbar(label='Magnitude')
        plt.title('Wavelet Transform (CWT) of the Audio Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Scale')
        plt.yscale('log')
        plt.ylim(scales[-1], scales[0])
        plt.show()
        
        peak_frequencies = freq[np.argmax(np.abs(coef), axis=0)]
        times = np.linspace(0, dur, len(peak_frequencies))

        plt.figure(figsize=(12, 8))
        plt.plot(times, peak_frequencies, label='Peak Frequency')
        plt.ylim(512, 10000)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'Peak Frequency Over Time for First {dur} Seconds')
        plt.legend()
        plt.show()

# %%
# response plots for applied filters
def filter_response_plots(y, sr, freq):
    b, a = get_high_pass_filter(freq, sr_ft)
    w, h = scipy.signal.freqz(b, a)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(0.5 * sr * w / np.pi, np.abs(h), 'b')
    plt.title('High-Pass Filter Frequency Response - Gain')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(0.5 * sr * w / np.pi, np.angle(h), 'b')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid()

    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    magnitude_dB = 20 * np.log10(np.abs(h))  # Convert magnitude to dB
    plt.plot(0.5 * sr * w / np.pi, magnitude_dB, 'b')
    plt.title('High-Pass Filter Frequency Response - dB')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(0.5 * sr * w / np.pi, np.angle(h), 'b')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid()

    plt.tight_layout()
    plt.show()

# %%
# produce_spectrogram(y, sr, 900, method='stft', dur=20)

# %%
# produce_spectrogram(y_ft, sr_ft, 900, method='wavelet', dur=-1)

# %%


# %%
def detect_energy_peaks(S_filtered, sr, height, distance, amp_min=80, amp_max=320, wavelet=False):
    if wavelet:
        energy_over_time = S_filtered  # For wavelet, S_filtered is 1D (total_power)

        print(f"Energy over time shape: {energy_over_time.shape}, dtype: {energy_over_time.dtype}")
        print(f"Energy over time (snippet): {energy_over_time[:10]}")

        # Detect peaks in the wavelet energy signal
        peaks, _ = scipy.signal.find_peaks(energy_over_time, height=height, distance=distance)
        peaks = [p for p in peaks if amp_min <= energy_over_time[p] <= amp_max]

        # Calculate baseline energy between peaks
        baseline_energy = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            baseline_seg = energy_over_time[start_idx: end_idx]
            baseline_val = baseline_seg[baseline_seg < np.median(baseline_seg)]
            baseline_energy.extend(baseline_val)

        # Calculate amplitude threshold
        amp_threshold = np.mean(baseline_energy) + 3 * np.std(baseline_energy)
        print(f'mean: {np.mean(baseline_energy)}')
        print(f'amp threshold: {amp_threshold}')

        # Map indices to time
        total_duration = len(energy_over_time) / sr
        full_times = np.linspace(0, total_duration, len(energy_over_time))
        times = np.array(peaks) / sr  # Convert peak indices to time

        # Plotting energy over time with peaks
        plt.figure(figsize=(10, 6))
        plt.plot(full_times, energy_over_time, label='Energy Over Time (Raw Amplitude)')
        plt.scatter(times, energy_over_time[peaks], color='red', label='Detected Peaks')

        # Define time segments based on amplitude threshold
        time_segments = []
        start_end_times = []

        for peak_idx in peaks:
            start_idx = peak_idx
            while start_idx > 0 and energy_over_time[start_idx] > amp_threshold:
                start_idx -= 1

            end_idx = peak_idx
            while end_idx < len(energy_over_time) - 1 and energy_over_time[end_idx] > amp_threshold:
                end_idx += 1

            start_time = start_idx / sr
            end_time = end_idx / sr

            if start_time < end_time:
                time_segments.append((start_time, end_time))
                start_end_times.append((start_time, end_time))

        # Highlight detected intervals on the plot
        for (start_time, end_time) in start_end_times:
            plt.fill_between(full_times, energy_over_time, where=((full_times >= start_time) & (full_times <= end_time)),
                             color='yellow', alpha=0.3, label='Detected Interval' if start_time == start_end_times[0][0] else "")

        plt.xlabel('Time (s)')
        plt.ylabel('Energy (Wavelet Power)')
        plt.title('Wavelet Energy Over Time with Detected Peaks and Intervals')
        plt.legend()
        plt.show()

        print(f'# of snippets: {len(time_segments)}')

        return time_segments, energy_over_time
    else:
        # similar ideas for stft transform
        energy_over_time = np.sum(S_filtered, axis=0)
    
        print(f"Energy over time shape: {energy_over_time.shape}, dtype: {energy_over_time.dtype}")
        print(f"Energy over time (snippet): {energy_over_time[:10]}")

        peaks, _ = scipy.signal.find_peaks(energy_over_time, height=height, distance=distance)

        peaks = [p for p in peaks if amp_min <= energy_over_time[p] <= amp_max]

        baseline_energy = []
        for i in range(len(peaks) - 1):
            start_idx = peaks[i]
            end_idx = peaks[i + 1]
            baseline_seg = energy_over_time[start_idx: end_idx]

            baseline_val = baseline_seg[baseline_seg < np.median(baseline_seg)]
            baseline_energy.extend(baseline_val)

        amp_threshold = np.mean(baseline_energy) + 3 * np.std(baseline_energy)

        print(f'mean: {np.mean(baseline_energy)}')
        print(f'amp threshold: {amp_threshold}')

        times = librosa.frames_to_time(peaks, sr=sr)
        full_times = librosa.frames_to_time(np.arange(S_filtered.shape[1]), sr=sr, hop_length=512)

        plt.figure(figsize=(10, 6))
        plt.plot(full_times, energy_over_time, label='Energy Over Time (Raw Amplitude)')
        plt.scatter(times, energy_over_time[peaks], color='red', label='Detected Peaks')

        time_segments = []
        start_end_times = []

        for peak_idx in peaks:
            start_idx = peak_idx
            while start_idx > 0 and energy_over_time[start_idx] > amp_threshold:
                start_idx -= 1

            end_idx = peak_idx
            while end_idx < len(energy_over_time) - 1 and energy_over_time[end_idx] > amp_threshold:
                end_idx += 1

            start_time = librosa.frames_to_time(start_idx, sr=sr, hop_length=512)
            end_time = librosa.frames_to_time(end_idx, sr=sr, hop_length=512)

            if start_time < end_time:
                time_segments.append((start_time, end_time))
                start_end_times.append((start_time, end_time))

        for (start_time, end_time) in start_end_times:
            plt.fill_between(full_times, energy_over_time, where=((full_times >= start_time) & (full_times <= end_time)),
                             color='yellow', alpha=0.3, label='Detected Interval' if start_time == start_end_times[0][0] else "")

        plt.xlabel('Time (s)')
        plt.ylabel('Energy (Sum of Raw Amplitudes)')
        plt.title('Energy Over Time with Detected Peaks and Trimmed Intervals')
        plt.legend()
        plt.show()

        print(f'# of snippets: {len(time_segments)}')

        return time_segments, energy_over_time

def decompose_freq(file_path, y, sr, height, distance, get_avg=True, time_seg_dur=0.2, dur=20, amp_min=80, amp_max=320):
    if dur != -1:
        y = y[:int(dur * sr)]
      
    # stft transformation
    D_filtered = librosa.stft(y, hop_length=1024)
    S_filtered = np.abs(D_filtered)
        
    time_segments, energy_over_time = detect_energy_peaks(S_filtered, sr, height=height, distance=distance, amp_min=amp_min, amp_max=amp_max)
    
    # plt.figure(figsize=(10, 6))
    segments_data = []
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=D_filtered.shape[0] * 2 - 1)

    # get audio snippets based on start and end indices
    for start_time, end_time in time_segments:
        start_idx = librosa.time_to_frames(start_time, sr=sr)
        end_idx = librosa.time_to_frames(end_time, sr=sr)
        
        segment_slice = S_filtered[:, start_idx:end_idx]
        if get_avg:
            avg_spectrum = np.mean(segment_slice, axis=1)

            plt.plot(frequencies, avg_spectrum, label=f"Time {start_time:.2f}-{end_time:.2f}s")
            # maintain information
            segments_data.append({
                'spectrum': avg_spectrum,
                'start_time': start_time,
                'end_time': end_time,
                'file_name': file_path
            })
            
            plt.ylim(0, None)
            plt.xlim(512, sr // 2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (Raw)')
            plt.title('Frequency Decomposition for All Time Intervals')

            plt.legend()
            plt.show()
        else:
            segments_data.append({
                'spectrum': segment_slice,
                'start_time': start_time,
                'end_time': end_time,
                'file_name': file_path
            })
#             plt.imshow(
#                 segment_slice,
#                 aspect='auto',
#                 extent=[start_time, end_time, frequencies[0], frequencies[-1]],
#                 origin='lower',
#                 cmap='viridis'
#             )
#             plt.colorbar(label='Amplitude')
#             plt.xlabel('Time (s)')
#             plt.ylabel('Frequency (Hz)')
#             plt.title(f"Spectrogram for Time {start_time:.2f}-{end_time:.2f}s")
#             plt.show()
    
    return segments_data, frequencies


def process_audio_and_extract_segments(file_path, get_avg=True, dur=-1, height=50, distance=25, amp_min=80, amp_max=320):
    y, sr = librosa.load(file_path)
    
    y_filtered = band_stop_filter(y, sr, highcut=900)
    
    return decompose_freq(file_path, y_filtered, sr, dur=dur, height=height, distance=distance, get_avg=get_avg, amp_min=amp_min, amp_max=amp_max)

# %%
process_audio_and_extract_segments(file_path)

# %%
def generate_pairs(n):
    pairs = list(itertools.combinations(range(n), 2))
    return pairs

# PCA
def perform_pca_on_segments(y, sr, time_segments, n_comp=4):               
    data_matrix = np.array(data_matrix)
    
    scaler = StandardScaler()
    data_matrix_std = scaler.fit_transform(data_matrix)
    
    pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(data_matrix_std)
    
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance by the components: {explained_variance}')

    component_pairs = generate_pairs(n_comp)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for ax, (i, j) in zip(axes.flat, component_pairs):
        ax.scatter(pca_result[:, i], pca_result[:, j], c='blue', marker='o')
        ax.set_title(f'PCA Components {i+1}-{j+1}')
        ax.set_xlabel(f'PC {i+1} ({explained_variance[i]*100:.2f}% variance)')
        ax.set_ylabel(f'PC {j+1} ({explained_variance[j]*100:.2f}% variance)')

    plt.tight_layout()
    plt.show()

# %%
# perform_pca_on_segments(y, sr, time_segments)

# %%
def perform_pca_on_segments(all_segments, n_components=4):
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(all_segments)
    
    print(f"Explained variance by the components: {pca.explained_variance_ratio_}")
    
    return pca_transformed, pca

def plot_pca_results(pca_transformed, n_components):
    pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)]
    
    plt.figure(figsize=(12, 8))
    
    for (i, (comp1, comp2)) in enumerate(pairs):
        plt.subplot(2, 3, i+1)
        plt.scatter(pca_transformed[:, comp1], pca_transformed[:, comp2], alpha=0.6, edgecolors='w', s=50)
        plt.title(f'PCA Components {comp1+1} vs {comp2+1}')
        plt.xlabel(f'Component {comp1+1}')
        plt.ylabel(f'Component {comp2+1}')
    
    plt.tight_layout()
    plt.show()

# %%
folder_path = '../audios/lab_audios/first_batch/'
file_names = [f'recording_20240927_204919-{i:02d}.wav' for i in range(1, 17)]
all_segments = []
all_frequencies = []
# for f in file_names:
#     file_path = os.path.join(folder_path, f)
#     print(f'Processing file: {file_path}')
#     segments, freqs = process_audio_and_extract_segments(file_path, get_avg=False)
#     all_segments.extend(segments)
#     all_frequencies.append(freqs)
    
# with open(f'{folder_path}all_segments.pkl', 'wb') as file:
#     pickle.dump(all_segments, file)

# print(f'All segments have been saved to all_segments.pkl')

# with open(f'{folder_path}all_segments_raw.pkl', 'wb') as file:
#     pickle.dump(all_segments, file)

# print(f'All segments have been saved to all_segments_raw.pkl')

# with open(f'{folder_path}all_frequencies.pkl', 'wb') as file:
#     pickle.dump(all_frequencies, file)

# print(f'All segments have been saved to all_frequencies.pkl')

with open(f'{folder_path}all_segments.pkl', 'rb') as file:
    all_segments = pickle.load(file)

# with open(f'{folder_path}all_segments_raw.pkl', 'rb') as file:
#     all_segments = pickle.load(file)
    
# with open(f'{folder_path}all_frequencies.pkl', 'rb') as file:
#     all_frequencies = pickle.load(file)

print(all_segments[0])

spectrums = np.array([s['spectrum'] for s in all_segments])
pca_transformed, pca = perform_pca_on_segments(spectrums)
plot_pca_results(pca_transformed, n_components=4)

# %%
def process_time_frames(all_segments):
    frames = []
    signal_indices = []

    print(f"Processing {len(all_segments)} signals...")

    for idx, segment in enumerate(all_segments):
        spectrum = segment['spectrum']
        num_frames = spectrum.shape[1]
        
        print(f"Processing signal {idx + 1}/{len(all_segments)}: {num_frames} frames")
        
        frames.extend(spectrum.T)
        signal_indices.extend([idx] * num_frames)
        
        if (idx + 1) % 10 == 0 or idx == len(all_segments) - 1:
            print(f"{idx + 1}/{len(all_segments)} signals processed...")

    frames = np.array(frames)
    
    print(f"Processing complete. Extracted {len(frames)} frames in total.")
    return frames, signal_indices

def cluster_time_frames(frames, n_clusters):
    scaler = StandardScaler()
    frames_scaled = scaler.fit_transform(frames)
    
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    frame_labels = spectral.fit_predict(frames_scaled)
    return frame_labels

def reconstruct_signals(signal_indices, frame_labels):
    from collections import defaultdict
    signal_clusters = defaultdict(list)
    
    for idx, label in zip(signal_indices, frame_labels):
        signal_clusters[idx].append(label)
    
    reconstructed_signals = {k: np.array(v) for k, v in signal_clusters.items()}
    return reconstructed_signals

def secondary_clustering(reconstructed_signals, n_clusters):
    signals = list(reconstructed_signals.values())
    signals_padded = pad_signals_to_uniform_length(signals)  # Ensure uniform length
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(signals_padded)
    
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    signal_labels = spectral.fit_predict(signals_scaled)
    return signal_labels

# %%
# frames, signal_indices = process_time_frames(all_segments)
# frame_labels = cluster_time_frames(frames, n_clusters=10)
# reconstructed_signals = reconstruct_signals(signal_indices, frame_labels)
# signal_labels = secondary_clustering(reconstructed_signals, n_clusters=5)

# # Analyze results
# for idx, label in enumerate(signal_labels):
#     print(f"Signal {idx}: Cluster {label}")

# %%
# duration
durations = [s['end_time'] - s['start_time'] for s in all_segments]
d_range = max(durations) - min(durations)
n_bins = 30

plt.figure(figsize=(10, 6))
plt.hist(durations, bins=n_bins, edgecolor='black', alpha=0.7)
plt.title("Histogram of Spectrum Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(f'Bin size: {d_range / n_bins}')

# resolution
resolutions = []
for freq in all_frequencies:
    min_resolution = np.min(np.diff(freq))
    resolutions.append(min_resolution)
    
plt.figure(figsize=(10, 6))
plt.hist(resolutions, bins=50, alpha=0.7, edgecolor='black')
plt.title("Histogram of Frequency Resolutions")
plt.xlabel("Resolution (Hz)")
plt.ylabel("Frequency Count")
plt.grid(axis='y', alpha=0.75)
plt.show()

print(resolutions)

# %%
def run_kmeans_full(data, n_clusters=4):
    spectrums = np.array([s['spectrum'] for s in all_segments])
    kmeans_full = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_full = kmeans_full.fit_predict(spectrums)
    
    for i, label in enumerate(labels_full):
        file_name = data[i]['file_name'][-6:-4]
        start_t = data[i]['start_time']
        end_t = data[i]['end_time']
        data[i]['cluster'] = label
        print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
    return labels_full

# %%
n_clusters = 5
labels_full = run_kmeans_full(all_segments, n_clusters)

# %%
# provide a more straightforward output
def group_segments_by_cluster(all_segments, n_clusters):
    clusters = [[] for _ in range(n_clusters)]
    for segment in all_segments:
        cluster_label = segment['cluster']
        clusters[cluster_label].append(segment)
    return clusters

# put detected audio snippets together
def concatenate_audio_snippets(cluster_segments, silence_duration=750):
    combined_audio = AudioSegment.silent(duration=0)
    silence = AudioSegment.silent(duration=silence_duration)
    cnt = 0
    for segment in cluster_segments:
        cnt += 1
        print(f'constantly processing segments{cnt}')
        audio_snippet = AudioSegment.from_wav(segment['file_name'])
        start_time = segment['start_time'] * 1000
        end_time = segment['end_time'] * 1000
        
        snippet = audio_snippet[start_time:end_time]
        
        combined_audio += snippet + silence
    
    return combined_audio

# write to file
def save_combined_audio(combined_audio, output_path):
    combined_audio.export(output_path, format="wav")
    print(f"Combined audio saved to {output_path}")

# %%
def combine_audio(all_segments, n_clusters, suffix):
    clustered_segments = group_segments_by_cluster(all_segments, n_clusters)

#     for cluster_index, cluster_segments in enumerate(clustered_segments):
#         if cluster_segments:
#             print(len(cluster_segments))
#             combined_audio = concatenate_audio_snippets(cluster_segments)
#             output_path = f'../audios/lab_audios/first_batch/combined_audio_cluster_{cluster_index}{suffix}.wav'
#             save_combined_audio(combined_audio, output_path)

# %%
# compute a average spectrum
def compute_cluster_averages_and_deltas(data, labels, n_clusters):
    cluster_averages = {}
    cluster_deltas = {i: [] for i in range(n_clusters)}
    
    for cluster_id in range(n_clusters):
        cluster_spectra = np.array([s['spectrum'] for s, label in zip(data, labels) if label == cluster_id])
        
        if len(cluster_spectra) > 0:
            average_spectrum = np.mean(cluster_spectra, axis=0)
            cluster_averages[cluster_id] = average_spectrum
            
            for spectrum in cluster_spectra:
                delta = spectrum - average_spectrum
                cluster_deltas[cluster_id].append(delta)
                
    for i, label in enumerate(labels):
        data[i]['delta'] = cluster_deltas[label][i % len(cluster_deltas[label])]
    
    return cluster_averages, cluster_deltas

# plot result
def plot_cluster_spectra_with_average(cluster_averages, cluster_deltas, n_clusters):
    for cluster_id in range(n_clusters):
        plt.figure(figsize=(10, 6))
        
        for delta in cluster_deltas[cluster_id]:
            spectrum = cluster_averages[cluster_id] + delta
            plt.plot(spectrum, color='blue', alpha=0.3)  # Individual spectra with transparency
        
        plt.plot(cluster_averages[cluster_id], color='red', linewidth=2, label='Average Spectrum')
        
        plt.title(f'Cluster {cluster_id} Spectra with Average Spectrum')
        plt.xlabel('Frequency Bin')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()
    
# compute standard deviation
def compute_std_per_frequency_in_clusters(cluster_deltas):
    cluster_frequency_std_dev = {}

    for cluster_id, deltas in cluster_deltas.items():
        deltas_array = np.array(deltas)

        std_dev_per_frequency = np.std(deltas_array, axis=0)
        
        cluster_frequency_std_dev[cluster_id] = std_dev_per_frequency
        
    for cluster_id, std_dev in cluster_frequency_std_dev.items():
        print(f"Cluster {cluster_id} - Standard Deviation per Frequency Bin:", std_dev)
    
    return cluster_frequency_std_dev

# total variance as a quality metric
def calculate_total_variance(cluster_deltas):
    total_var = 0
    total_points = 0
    
    for cluster_id, deltas in cluster_deltas.items():
        deltas_array = np.array(deltas)
        n_points = deltas_array.shape[0]
        
        if n_points > 1:
            cluster_variance = np.sum(np.var(deltas_array, axis=0))
            total_var += cluster_variance * n_points
            total_points += n_points
        elif n_points == 1:
            total_points += 1
    mean_variance = total_var / total_points if total_points > 0 else 0
    return mean_variance

# %%
combine_audio(all_segments, n_clusters, suffix='')

# %%
# view clustering result after pca
def plot_pca_clusters(pca_transformed, labels, n_components=4):
    """Plot PCA-transformed data with color-coding based on cluster labels."""
    plt.figure(figsize=(12, 8))
    
    # Define a color map to distinguish clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # 'tab10' colormap for 10 distinct colors
    
    # Create pairwise PCA component plots (e.g., 1 vs 2, 2 vs 3, etc.)
    component_pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)]
    
    for idx, (i, j) in enumerate(component_pairs):
        plt.subplot(2, 3, idx + 1)
        for label in unique_labels:
            cluster_data = pca_transformed[labels == label]  # Data points in this cluster
            plt.scatter(cluster_data[:, i], cluster_data[:, j], label=f"Cluster {label}", alpha=0.6, cmap='tab10')
        plt.xlabel(f'Component {i+1}')
        plt.ylabel(f'Component {j+1}')
        plt.title(f'PCA Components {i+1} vs {j+1}')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
plot_pca_clusters(pca_transformed, labels_full)

# %%
# plot explained variance against number of principal components
def plot_explained_variance(spectrums, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(spectrums)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.6, align='center',
            label='Individual Explained Variance')
    plt.step(range(1, n_components + 1), np.cumsum(explained_variance_ratio), where='mid',
             label='Cumulative Explained Variance', color='red')
    
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Explained Variance by Principal Components (up to {n_components} PCs)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    return np.cumsum(explained_variance_ratio)

# %%
plot_explained_variance(spectrums, n_components=38)

# %%
# PCA-Kmeans
def perform_pca_and_kmeans(data, n_components, n_clusters=4, pca=True):
    spectrums = np.array([s['spectrum'] for s in all_segments])
    if pca:
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(spectrums)
    else:
        pca = None
        pca_transformed = spectrums
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_transformed)
    
    cluster_sizes = np.bincount(labels)
    
    sorted_cluster_indices = np.argsort(-cluster_sizes)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_indices)}
    
    reordered_labels = np.array([label_mapping[label] for label in labels])
    
    for i, label in enumerate(reordered_labels):
        file_name = data[i]['file_name'][-6:-4]
        start_t = data[i]['start_time']
        end_t = data[i]['end_time']
        data[i]['cluster'] = label
        # print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
    
    return pca_transformed, reordered_labels, pca

# %%
n_clusters = 5
pca_transformed_90, labels_90, pca_90 = perform_pca_and_kmeans(all_segments, n_components=38, n_clusters=n_clusters)

combine_audio(all_segments, n_clusters, '-90pcaEd')

# %%
pca_transformed_80, labels_80, pca_80 = perform_pca_and_kmeans(all_segments, n_components=13, n_clusters=n_clusters)

combine_audio(all_segments, n_clusters, '-80pcaEd')

# %%
_, labels_full, _ = perform_pca_and_kmeans(all_segments, n_components=38, n_clusters=n_clusters, pca=False)

combine_audio(all_segments, n_clusters, 'full')
cluster_avg_k, cluster_deltas_k = compute_cluster_averages_and_deltas(all_segments, labels_full, n_clusters)
plot_cluster_spectra_with_average(cluster_avg_k, cluster_deltas_k, n_clusters)

std_sc = compute_std_per_frequency_in_clusters(cluster_deltas_k)

# %%
# see how clusters are different from each variance explained threshold
def compare_clusters(labels1, labels2, data, name1="90%", name2="full"):
    n_clusters = max(np.max(labels1), np.max(labels2)) + 1
    comparisons = {}

    for cluster in range(n_clusters):
        indices_1 = set(np.where(labels1 == cluster)[0])
        indices_2 = set(np.where(labels2 == cluster)[0])

        common_points = indices_1.intersection(indices_2)
        unique_to_1 = indices_1 - indices_2
        unique_to_2 = indices_2 - indices_1

        comparisons[cluster] = {
            "common": common_points,
            "unique_to_1": unique_to_1,
            "unique_to_2": unique_to_2,
        }

        print(f"\nComparison for Cluster {cluster} ({name1} vs {name2}):")
        # print(f"Common data points ({len(common_points)}): {sorted(common_points)}")
        print(f"Unique to {name1} ({len(unique_to_1)}): {sorted(unique_to_1)}")
        print(f"Unique to {name2} ({len(unique_to_2)}): {sorted(unique_to_2)}")

    return comparisons


comp_90_full = compare_clusters(labels_90, labels_full, all_segments, name1="90%", name2="full")

comp_80_full = compare_clusters(labels_80, labels_full, all_segments, name1="80%", name2="full")


# %%
# dbscan
def perform_pca_and_dbscan(data, n_components=38, eps=0.5, min_samples=5, pca=False):
    spectrums = np.array([s['spectrum'] for s in data])
    if pca:
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(spectrums)
    else:
        pca = None
        pca_transformed = spectrums
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pca_transformed)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels[sorted_indices])}
    
    reordered_labels = np.array([label_mapping[label] if label != -1 else -1 for label in labels])
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    
    for i, label in enumerate(reordered_labels):
        file_name = data[i]['file_name'][-6:-4]
        start_t = data[i]['start_time']
        end_t = data[i]['end_time']
        data[i]['cluster'] = label
        # print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
        
    return pca_transformed, reordered_labels, pca, n_clusters

# %%
pca_transformed, labels_dbscan, pca_model, n_clusters = perform_pca_and_dbscan(all_segments, eps=1, min_samples=5)
print(n_clusters)
combine_audio(all_segments, n_clusters+1, '-DBSCAN')

# %%
# define some hyper-features
def extract_broad_features(data):
    broad_features = []
    for segment in data:
        spectrum = segment['spectrum']
        duration = segment['end_time'] - segment['start_time']
        frequency_range = np.max(spectrum) - np.min(spectrum)
        average_frequency = np.mean(spectrum)
        energy = np.sum(spectrum)
        
#         broad_features.append([frequency_range, duration, average_frequency, energy])
        broad_features.append([frequency_range, average_frequency, energy])
    return np.array(broad_features)

def hierarchical_clustering(features, distance_threshold, method='ward'):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    Z = linkage(features, method=method)
    labels = fcluster(Z, t=distance_threshold, criterion='distance')
    return labels, Z

# two-step hierarchical clustering
def two_step_hierarchical_clustering(data, broad_distance, refined_distance):
    broad_features = extract_broad_features(data)
    initial_labels, broad_Z = hierarchical_clustering(broad_features, distance_threshold=broad_distance)
    
    refined_labels = np.zeros(len(data), dtype=int)
    next_label = 1
    
    for cluster_id in np.unique(initial_labels):
        cluster_indices = np.where(initial_labels == cluster_id)[0]
        cluster_data = [data[i] for i in cluster_indices]
        
        detailed_features = np.array([segment['spectrum'] for segment in cluster_data])
        detailed_labels, refined_Z = hierarchical_clustering(detailed_features, distance_threshold=refined_distance)
        
        for i, index in enumerate(cluster_indices):
            refined_labels[index] = next_label + detailed_labels[i] - 1
        next_label += np.max(detailed_labels)
    refined_labels -= 1
    
    return refined_labels, broad_Z, refined_Z, len(np.unique(refined_labels))

def perform_hierarchical_clustering(data, method='ward', distance_threshold=None, plot_dendrogram=True):
    spectrums = np.array([s['spectrum'] for s in data])
    scaler = StandardScaler()
    spectrums = scaler.fit_transform(spectrums)
    
    Z = linkage(spectrums, method=method)
    
    if plot_dendrogram:
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Data Points")
        plt.ylabel("Distance")
        if distance_threshold is not None:
            plt.axhline(y=distance_threshold, color='r', linestyle='--', label=f'Distance Threshold ({distance_threshold})')
            plt.legend()
        plt.show()
    
    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion='distance')
        labels -= 1
        num_clusters = len(np.unique(labels))
    else:
        raise ValueError("Please specify a distance_threshold to form clusters.")
        
    for i, label in enumerate(labels):
        file_name = data[i]['file_name'][-6:-4]
        start_t = data[i]['start_time']
        end_t = data[i]['end_time']
        data[i]['cluster'] = label
        # print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
        
    return labels, num_clusters

# %%
# generic hierarchical clustering
distance_threshold = 250
labels_hierarchical, n_clusters = perform_hierarchical_clustering(all_segments, method='ward', distance_threshold=distance_threshold)
combine_audio(all_segments, n_clusters, '-hierarical')
cluster_avg_h, cluster_deltas_h = compute_cluster_averages_and_deltas(all_segments, labels_hierarchical, n_clusters)
plot_cluster_spectra_with_average(cluster_avg_h, cluster_deltas_h, n_clusters)

std_h= compute_std_per_frequency_in_clusters(cluster_deltas_h)

# %%
# two-step hierarchical clustering
broad_distance = 300
refined_distance = 75
final_labels, broad_Z, refined_Z, n_clusters = two_step_hierarchical_clustering(all_segments, broad_distance, refined_distance)
print(n_clusters)

cluster_avg_h2, cluster_deltas_h2 = compute_cluster_averages_and_deltas(all_segments, final_labels, n_clusters)
plot_cluster_spectra_with_average(cluster_avg_h2, cluster_deltas_h2, n_clusters)

std_h2 = compute_std_per_frequency_in_clusters(cluster_deltas_h2)

# %%
# filter out segments that formed their own clusters
def filter_segments(data, labels, min_cluster_size=2):
    cluster_counts = Counter(labels)
    all_segments_filtered = []
    discarded_segments = []
    for i, label in enumerate(labels):
        if cluster_counts[label] >= min_cluster_size:
            all_segments_filtered.append(data[i])
        else:
            discarded_segments.append(data[i])
    return all_segments_filtered, discarded_segments

# %%
filtered_all_segments, discarded_segments = filter_segments(all_segments, final_labels)
print(len(all_segments))
print(len(filtered_all_segments))

# %%
# print('Process discarded segments...')
# output_path = '../audios/lab_audios/first_batch/2stepHierarchicalGarbage-300-75.wav'
# combined_audio = concatenate_audio_snippets(discarded_segments, silence_duration=750)
# save_combined_audio(combined_audio, output_path)

# %%
# clustering quality metrics
def compute_normalized_cut(affinity_matrix, labels):
    unique_labels = np.unique(labels)
    total_ncut = 0

    for label in unique_labels:
        cluster = np.where(labels == label)[0]
        complement = np.where(labels != label)[0]

        cut = np.sum(affinity_matrix[np.ix_(cluster, complement)])
        assoc = np.sum(affinity_matrix[cluster, :])

        if assoc > 0:
            ncut = cut / assoc
            total_ncut += ncut

    return total_ncut

def compute_modularity(affinity_matrix, labels):
    graph = nx.from_numpy_array(affinity_matrix)

    communities = {label: [] for label in np.unique(labels)}
    for node, label in enumerate(labels):
        communities[label].append(node)
    
    community_list = list(communities.values())
    
    modularity_score = nx.algorithms.community.quality.modularity(graph, community_list)
    return modularity_score



def perform_spectral_clustering(data, n_clusters, affinity='nearest_neighbors'):
    spectrums = np.array([s['spectrum'] for s in data])
#     if pca:
#         pca_transformer = PCA(n_components=n_components)
#     else:
#         pca_transformer = KernelPCA(n_components=n_components, kernel='rbf')
        
#     pca_transformed = pca_transformer.fit_transform(spectrums)
    scaler = StandardScaler()
    spectrums = scaler.fit_transform(spectrums)
    
#     if affinity == 'rbf' and gamma is not None:
#         affinity_matrix = rbf_kernel(spectrums, gamma=gamma)
#     elif affinity == 'nearest_neighbors':
#         affinity_matrix = kneighbors_graph(spectrums, n_neighbors=n_clusters, include_self=True).toarray()
#     else:
#         raise ValueError("Unsupported affinity type or missing parameters.")
    
    spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
    labels = spectral.fit_predict(spectrums)
    
    num_clusters = len(np.unique(labels))
    print(f'Number of clusters: {num_clusters}')
    
    
#     normalized_cut = compute_normalized_cut(affinity_matrix, labels)
#     print(f"Normalized Cut Score: {normalized_cut}")
    
    for i, label in enumerate(labels):
#         file_name = data[i]['file_name'][-6:-4]
#         start_t = data[i]['start_time']
#         end_t = data[i]['end_time']
        data[i]['cluster'] = label
        # print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
    
#     if plot_clusters and spectrums.shape[1] == 2:
#         plt.figure(figsize=(8, 6))
#         plt.scatter(spectrums[:, 0], spectrums[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
#         plt.title("Spectral Clustering")
#         plt.xlabel("Component 1")
#         plt.ylabel("Component 2")
#         plt.colorbar(label="Cluster Label")
#         plt.show()
    
    return labels



# %%
# labels_sc = perform_spectral_clustering(all_segments, n_clusters)
# combine_audio(all_segments, n_clusters, '-sc')
# cluster_avg_sc, cluster_deltas_sc_13 = compute_cluster_averages_and_deltas(all_segments, labels_sc, n_clusters)
# plot_cluster_spectra_with_average(cluster_avg_sc, cluster_deltas_sc_13, n_clusters)

# std_sc_13 = compute_std_per_frequency_in_clusters(cluster_deltas_sc_13)

# %%
# another clustering quality metric
def calculate_silhouette_score(data, labels):
    spectrums = np.array([s['spectrum'] for s in data])
    score = silhouette_score(spectrums, labels, metric='euclidean')
    return score

# define a cost for evaluating clustering results
def evaluate_cost(data, distance_threshold, penalty_w):
    labels_hierarchical, n_clusters = perform_hierarchical_clustering(data, method='ward', distance_threshold=distance_threshold)
    combine_audio(all_segments, n_clusters, '-hierarical')
    cluster_avg_h, cluster_deltas_h = compute_cluster_averages_and_deltas(data, labels_hierarchical, n_clusters)
    std_h = compute_std_per_frequency_in_clusters(cluster_deltas_h)
    variance_term = calculate_total_variance(cluster_deltas_h)
    # variance_term = 1 - calculate_silhouette_score(data, labels_hierarchical)

    num_clusters = len(cluster_deltas_h)
    penalty_term = penalty_weight * num_clusters
    print(f'variance term: {variance_term}')
    
    cost = variance_term + penalty_term
    return cost, num_clusters

# %%
# penalty_weight = 1e-3
# distance_thresholds = np.arange(10, 401, step=10)
# best_cost = float('inf')
# best_threshold = None
# best_num_clusters = None

# for distance in distance_thresholds:
#     cost, num_clusters = evaluate_cost(all_segments, distance, penalty_weight)
#     print(f"Distance: {distance}, Cost: {cost}, Num Clusters: {num_clusters}")
#     if cost < best_cost:
#         best_cost = cost
#         best_threshold = distance
#         best_num_clusters = num_clusters

# print(f"Optimal distance threshold: {best_threshold} with cost {best_cost} and clusters: {best_num_clusters}")

# %%
# spectral clustering again
# labels_sc = perform_spectral_clustering(all_segments, n_clusters=14)
# combine_audio(all_segments, n_clusters=14, suffix='-sc')
# cluster_avg_sc, cluster_deltas_sc_14 = compute_cluster_averages_and_deltas(all_segments, labels_sc, n_clusters=14)
# plot_cluster_spectra_with_average(cluster_avg_sc, cluster_deltas_sc_14, n_clusters=14)

# std_sc_14 = compute_std_per_frequency_in_clusters(cluster_deltas_sc_14)

# labels_sc = perform_spectral_clustering(all_segments, n_clusters=16)
# combine_audio(all_segments, n_clusters=16, suffix='-sc')
# cluster_avg_sc, cluster_deltas_sc_16 = compute_cluster_averages_and_deltas(all_segments, labels_sc, n_clusters=16)
# plot_cluster_spectra_with_average(cluster_avg_sc, cluster_deltas_sc_16, n_clusters=16)

# std_sc_16 = compute_std_per_frequency_in_clusters(cluster_deltas_sc_16)


# labels_sc = perform_spectral_clustering(all_segments, n_clusters=50)
# combine_audio(all_segments, n_clusters=50, suffix='-sc')
# cluster_avg_sc, cluster_deltas_sc_50 = compute_cluster_averages_and_deltas(all_segments, labels_sc, n_clusters=50)
# plot_cluster_spectra_with_average(cluster_avg_sc, cluster_deltas_sc_50, n_clusters=50)

# std_sc_50 = compute_std_per_frequency_in_clusters(cluster_deltas_sc_50)

# %%
# total_variance_13 = calculate_total_variance(cluster_deltas_sc_13)
# total_variance_14 = calculate_total_variance(cluster_deltas_sc_14)
# total_variance_50 = calculate_total_variance(cluster_deltas_sc_50)


# print(f"Total Variance for 13 Clusters: {total_variance_13}")
# print(f"Total Variance for 14 Clusters: {total_variance_14}")
# print(f"Total Variance for 25 Clusters: {total_variance_50}")

# %%
# max_clusters = 30
# best_cost = float('inf')
# best_n = 0
# penalty = 1e-1
# for n_cluster in range(2, max_clusters+1):
#     labels_sc = perform_spectral_clustering(all_segments, n_clusters=n_cluster)
#     combine_audio(all_segments, n_clusters=n_cluster, suffix='-sc')
#     cluster_avg_sc, cluster_deltas_sc = compute_cluster_averages_and_deltas(all_segments, labels_sc, n_clusters=n_cluster)
#     std_sc = compute_std_per_frequency_in_clusters(cluster_deltas_sc)
#     variance_term = calculate_total_variance(cluster_deltas_sc)
#     # variance_term = 1 - calculate_silhouette_score(all_segments, labels_sc)
    
#     cost = variance_term + penalty * n_cluster
    
#     print(f"Cost: {cost}, Num Clusters: {n_cluster}")
    
#     if cost < best_cost:
#         best_cost = cost
#         best_n = n_cluster
# print(f"Best cost {best_cost} and clusters: {best_n}")

# %%
# Isomap
spectrums = np.array([s['spectrum'] for s in all_segments])
n_neighbors = 10
n_components = 10
n_clusters = 10

isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
reduced_features = isomap.fit_transform(spectrums)

spectral_clustering = SpectralClustering(
    n_clusters=n_clusters,
    affinity='nearest_neighbors',
    assign_labels='kmeans',
    random_state=42
)
labels = spectral_clustering.fit_predict(reduced_features)
# plt.figure(figsize=(10, 6))
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=50)
# plt.title('Spectral Clustering Results after Isomap')
# plt.xlabel('Isomap Dimension 1')
# plt.ylabel('Isomap Dimension 2')
# plt.colorbar(label='Cluster')
# plt.show()

for i, label in enumerate(labels):
#         file_name = data[i]['file_name'][-6:-4]
#         start_t = data[i]['start_time']
#         end_t = data[i]['end_time']
    all_segments[i]['cluster'] = label
        # print(f'File: {file_name}, {start_t} : {end_t}: Cluster {label}')
        
cluster_avg_iso, cluster_deltas_iso = compute_cluster_averages_and_deltas(all_segments, labels, n_clusters)
plot_cluster_spectra_with_average(cluster_avg_iso, cluster_deltas_iso, n_clusters)

# %%
# investigate optimal number of clusters
# gap statistic
def compute_gap_statistic(data, max_clusters=30):
    optimal_k = optimalK.OptimalK(parallel_backend='joblib')
    return optimal_k(data, cluster_array=np.arange(1, max_clusters + 1))

# %%
spectrums = np.array([s['spectrum'] for s in all_segments])
print(compute_gap_statistic(spectrums))

# %%
# elbow method
def plot_elbow_spectral(data, max_clusters=5):
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        spectral = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
        labels = spectral.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Elbow Method for Spectral Clustering (Silhouette Score)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

# %%
spectrums = np.array([s['spectrum'] for s in all_segments])
plot_elbow_spectral(spectrums)

# %%
pkl_output_path = '../audios/lab_audios/first_batch/200RandomSegments.pkl'
output_audio_path = '../audios/lab_audios/first_batch/200RandomSegments.wav'

if os.path.exists(pkl_output_path):
    print(f"Loading existing random segments from {pkl_output_path}")
    with open(pkl_output_path, 'rb') as pkl_file:
        random_segments = pickle.load(pkl_file)
else:
    print("Random segments .pkl file not found. Generating new random segments...")
    random_segments = random.sample(all_segments, 200)
    with open(pkl_output_path, 'wb') as pkl_file:
        pickle.dump(random_segments, pkl_file)
    print(f"Random segments saved to {pkl_output_path}")

# combined_audio = concatenate_audio_snippets(random_segments, silence_duration=750)
# save_combined_audio(combined_audio, output_audio_path)

# %%
# just plot some spectrums
def plot_segments_in_grid(random_segments, grid_shape=(20, 10)):
    num_segments = len(random_segments)
    fig, axes = plt.subplots(*grid_shape, figsize=(24, 48))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    for idx, segment in enumerate(random_segments):
        if idx >= grid_shape[0] * grid_shape[1]:
            print(f"Warning: More segments than grid spaces. Only the first {grid_shape[0] * grid_shape[1]} will be plotted.")
            break
        
        spectrum = segment['spectrum']
        frequency_bins = np.arange(len(spectrum))
        
        ax = axes[idx]
        ax.plot(frequency_bins, spectrum)
        ax.set_ylim(0, 2)
        ax.set_title(f"Segment {idx+1}", fontsize=8)
        ax.set_xlabel('Frequency Bin', fontsize=6)
        ax.set_ylabel('Amplitude', fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(True)
    
    for ax in axes[num_segments:]:
        ax.axis('off')
    
    plt.tight_layout(pad=2.5, h_pad=2, w_pad=2)
    plt.show()

# %%
plot_segments_in_grid(random_segments)

# %%
# manual 'ground truth'
random_c0 = [1,3,7,8]
random_c1 = [4,9,12,16,18,19,22,23,24,25,26,30,34,35,36,37,39,41,42,43,45,46,47,54,56,58,62,63,68,70,73,75,77,78,83,84,85,87,89,90,93,96,100,49,2,10,32,38]
random_c2 = [11,13,14,33,40,44,50,52,55,57,59,60,71,76,80,81,82,86,92,94,97,61,5,28,29,48,65,66,72,91]
random_c3 = [17,20,27,31,51,64,69,79,98]
random_c4 = [53,67,74,88]
random_c5 = [95,6,21,99]
random_c6 = [15]

random_c0 = [i - 1 for i in random_c0]
random_c1 = [i - 1 for i in random_c1]
random_c2 = [i - 1 for i in random_c2]
random_c3 = [i - 1 for i in random_c3]
random_c4 = [i - 1 for i in random_c4]
random_c5 = [i - 1 for i in random_c5]
random_c6 = [i - 1 for i in random_c6]

cluster_to_segments = {
    0: random_c0,
    1: random_c1,
    2: random_c2,
    3: random_c3,
    4: random_c4,
    5: random_c5,
    6: random_c6,
}

# output_dir = '../audios/lab_audios/first_batch/'
# for cluster_id, segment_indices in cluster_to_segments.items():
#     cluster_segments = [random_segments[i] for i in segment_indices]
#     combined_audio = concatenate_audio_snippets(cluster_segments, silence_duration=750)
#     output_path = os.path.join(output_dir, f'ManualLabelCluster_{cluster_id}_refined.wav')
#     combined_audio.export(output_path, format="wav")
#     print(f"Cluster {cluster_id} audio saved to {output_path}")

# %%
# visualize affinity matrix
def visualize_affinity_matrix(data, gamma=1.0):
    # Compute the affinity matrix using RBF kernel
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)
    affinity_matrix = rbf_kernel(data, gamma=gamma)
    
    # Plot the affinity matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(affinity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.title('Affinity Matrix')
    plt.xlabel('Data Points')
    plt.ylabel('Data Points')
    plt.show()

spectrums = np.array([s['spectrum'] for s in all_segments])
visualize_affinity_matrix(spectrums, gamma=0.025)

# %%
# tuning affinity matrix by tuning gamma
def perform_spectral_clustering_with_gamma(data, gamma, n_clusters):
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(data)

    spectrums = np.array([s['spectrum'] for s in data])

#     affinity_matrix = rbf_kernel(spectrums, gamma=gamma)
#     np.fill_diagonal(affinity_matrix, 0)
    affinity_matrix = kneighbors_graph(spectrums, n_neighbors=10, include_self=True).toarray()
    np.fill_diagonal(affinity_matrix, 0)

    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        assign_labels='kmeans'
    )
    labels = spectral_clustering.fit_predict(affinity_matrix)

    for i, label in enumerate(labels):
        data[i]['cluster'] = label
    silhouette_avg = silhouette_score(affinity_matrix, labels, metric='precomputed')
    print(f"Silhouette Score for gamma={gamma} and n_clusters={n_clusters}: {silhouette_avg}")
    print(labels)
    return labels, silhouette_avg

# %%
gamma = 0.01
n_clusters = 10
labels_sc_10, silhouette_avg = perform_spectral_clustering_with_gamma(all_segments, gamma, n_clusters)
combine_audio(all_segments, n_clusters, suffix='-sc_gamma_005')

print(f"Unique labels: {np.unique(labels_sc_10)}")
print(f"Expected number of clusters: {n_clusters}")

cluster_avg_sc, cluster_deltas_sc_10 = compute_cluster_averages_and_deltas(all_segments, labels_sc_10, n_clusters=10)
plot_cluster_spectra_with_average(cluster_avg_sc, cluster_deltas_sc_10, n_clusters=10)

# %%
# experiment with n_clusters parameter
n_clusters_low = 2
n_cluster_high = 5
spectrums = np.array([s['spectrum'] for s in all_segments])
ncut_values = []
mod_scores = []
db_indices = []
clusters_range = range(n_clusters_low, n_cluster_high + 1)
for n_clusters in clusters_range:
    affinity_matrix = kneighbors_graph(spectrums, n_neighbors=n_clusters, include_self=True).toarray()
    labels = perform_spectral_clustering(all_segments, n_clusters)
    normalized_cut = compute_normalized_cut(affinity_matrix, labels)
    ncut_values.append(normalized_cut)
    
    mod_score = compute_modularity(affinity_matrix, labels)
    mod_scores.append(mod_score)
    
    if n_clusters > 1:
        db_index = davies_bouldin_score(spectrums, labels)
        db_indices.append(db_index)
    
plt.figure(figsize=(10, 6))
plt.plot(clusters_range, ncut_values, marker='o')
plt.title("Normalized Cut vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Normalized Cut Value")
plt.grid()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(clusters_range, mod_scores, marker='o')
plt.title("Modularity vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Modulartity Score")
plt.grid()
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(clusters_range, db_indices, marker='o')
plt.title("DB Index vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("DB Index")
plt.grid()
plt.show()

# %%
# output_path = '../audios/lab_audios/first_batch/all_segments.wav'
# combined_audio = concatenate_audio_snippets(all_segments)


# %%
# save_combined_audio(combined_audio, output_path)

# %%
def split_audio(file_path, split_time, output_file_1, output_file_2):
    """
    Splits an audio file at a given time and saves the resulting segments.
    :param file_path: Path to the input audio file
    :param split_time: Time (in seconds) where the split should occur
    :param output_file_1: Path to save the first segment
    :param output_file_2: Path to save the second segment
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Convert split time to milliseconds
    split_time_ms = split_time * 1000
    
    # Split the audio
    segment_1 = audio[:split_time_ms]  # First segment
    segment_2 = audio[split_time_ms:]  # Second segment
    
    # Export the segments
    segment_1.export(output_file_1, format="wav")
    print(f"First segment saved to {output_file_1}")
    
    segment_2.export(output_file_2, format="wav")
    print(f"Second segment saved to {output_file_2}")

# Example usage
# file_path = '../audios/lab_audios/first_batch/all_segments.wav'
# split_time = (39 * 60) + 48  # Convert 39:48 to seconds
# output_file_1 = '../audios/lab_audios/first_batch/all_segments_0.wav'
# output_file_2 = '../audios/lab_audios/first_batch/all_segments_1.wav'

# split_audio(file_path, split_time, output_file_1, output_file_2)

# %%
# a lot of compromises were made due to the computation intensity introduced by wavelet
# results are thus not very pretty
def process_audio_and_extract_segments_wavelet(file_path, dur=-1, height=0.001, distance=1, amp_min=0.02, amp_max=0.6, wavelet='db4'):
    y, sr = librosa.load(file_path)
    
    y_filtered = band_stop_filter(y, sr, highcut=900)
    
    return decompose_wavelet_chunked(file_path, y_filtered, sr, wavelet, height=height, distance=distance, amp_min=amp_min, amp_max=amp_max, dur=dur)

def decompose_wavelet_chunked(
    file_path, y, sr, wavelet, height, distance, chunk_size=10, amp_min=80, amp_max=320, dur=-1
):
    """Performs DWT without chunking (chunking lines are commented out)."""
    # num_chunks = int(len(y) / (chunk_size * sr)) + 1  # Calculate the number of chunks
    
    if dur != -1:
        y = y[:int(dur * sr)]
    segments_data = []

    # for i in range(num_chunks):
    #     start_idx = i * chunk_size * sr
    #     end_idx = min((i + 1) * chunk_size * sr, len(y))
    #     y_chunk = y[start_idx:end_idx]
    #     if len(y_chunk) == 0:  # Skip empty chunks
    #         continue

    # Perform DWT on the entire signal
    coeffs = pywt.wavedec(y, wavelet=wavelet, mode='periodization', level=None)
    power_levels = [np.square(np.abs(c)) for c in coeffs]

#     # Perform CWT on the entire signal
#     scales = np.arange(1, 128)  # Define the scales; adjust the range as needed for your data
#     coeffs, freqs = pywt.cwt(y, scales, wavelet, sampling_period=1/sr)  # Compute CWT

#     # Compute power levels from the wavelet coefficients
#     power_levels = np.abs(coeffs) ** 2

    # Aggregate energy over all levels
    total_power = np.zeros(len(y))  # Initialize as zero for same length as the signal
    for c in power_levels:
        interpolated = np.interp(
            np.linspace(0, len(y), len(y)),
            np.linspace(0, len(c), len(c)),
            c,
        )
        total_power += interpolated

    # Debugging total_power
    print(f"Total_power shape: {total_power.shape}, dtype: {total_power.dtype}")
    print(f"Total power (snippet): {total_power[:10]}")

    total_power = total_power.flatten()
    if total_power.ndim != 1:
        raise ValueError(f"Expected total_power to be 1-D, got {total_power.ndim}-D array.")

    # Detect energy peaks over the entire signal
    time_segments, energy_over_time = detect_energy_peaks(
        total_power, sr, height=height, distance=distance, amp_min=amp_min, amp_max=amp_max, wavelet=True
    )

    # Generate segment data for the entire signal
    frequencies = np.linspace(0, sr // 2, len(total_power))
    for start_time, end_time in time_segments:
        avg_spectrum = total_power[int(start_time * sr): int(end_time * sr)].mean()

        segments_data.append({
            'spectrum': avg_spectrum,
            'start_time': start_time,
            'end_time': end_time,
            'file_name': file_path
        })

    return segments_data

# %%
def process_individual_snippets_with_wavelet(all_segments, wavelet='db4', sr=None):
    processed_segments = []
    for idx, segment in enumerate(all_segments):
        file_name = segment['file_name']
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        print(f"Processing snippet {idx + 1}/{len(all_segments)}: {file_name}, {start_time:.2f}s to {end_time:.2f}s")
        
        y, original_sr = librosa.load(file_name, sr=sr)
        
        start_idx = int(start_time * original_sr)
        end_idx = int(end_time * original_sr)
        snippet = y[start_idx:end_idx]
        
        coeffs = pywt.wavedec(snippet, wavelet=wavelet, mode='periodization')
        
        power_levels = [np.square(np.abs(c)) for c in coeffs]
        
        processed_segments.append({
            'file_name': file_name,
            'start_time': start_time,
            'end_time': end_time,
            'wavelet_coeffs': coeffs,
            'wavelet_power': power_levels,
            'frequencies': None
        })
    
    return processed_segments


# %%
processed_segments_wavelet = process_individual_snippets_with_wavelet(all_segments, wavelet='db4')

# %%
# folder_path = '../audios/lab_audios/first_batch/'
# with open(f'{folder_path}all_segments_wavelet.pkl', 'wb') as file:
#     pickle.dump(processed_segments_wavelet, file)

# print(f'All segments have been saved to all_segments_wavelet.pkl')

# %%
with open(f'{folder_path}all_segments_wavelet.pkl', 'rb') as file:
    processed_segments_wavelet = pickle.load(file)

# %%
print(processed_segments_wavelet[0])

# %%
plt.figure(figsize=(15, 10))

_, sr = librosa.load("../audios/lab_audios/first_batch/recording_20240927_204919-01.wav")

for i, segment in enumerate(processed_segments_wavelet[:10], start=1):
    wavelet_power = segment['wavelet_power']  # List of arrays (one per level)
    levels = len(wavelet_power)  # Number of decomposition levels
    sr = sr  # Replace with the actual sampling rate
    nyquist = sr / 2  # Nyquist frequency

    # Calculate frequency ranges for each level
    freq_bands = [(nyquist / (2 ** i), nyquist / (2 ** (i - 1))) for i in range(1, levels + 1)]
    center_frequencies = [np.mean(band) for band in freq_bands]

    # Average wavelet power over time for each level
    avg_power = [np.mean(power) for power in wavelet_power if len(power) > 0]

    if len(avg_power) != len(center_frequencies):
        print(f"Skipping Segment {i} due to mismatch in levels and frequency bands.")
        continue

    plt.subplot(2, 5, i)
    plt.plot(center_frequencies, avg_power, label=f"Segment {i}")
    plt.xscale('log')  # Log scale for frequency axis
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average Power")
    plt.title(f"Segment {i}")
    plt.grid(True)

plt.tight_layout()
plt.show()


# %%



