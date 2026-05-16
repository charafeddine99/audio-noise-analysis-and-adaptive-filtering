import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
from scipy.fft import fft, fftfreq

from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    iirnotch
)

import librosa
import librosa.display
import soundfile as sf

# =========================================================
# FILTER FUNCTIONS
# =========================================================

def apply_lowpass(data, sr, cutoff=4000, order=5):

    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype='low'
    )

    return filtfilt(b, a, data)

# =========================================================

def apply_highpass(data, sr, cutoff=80, order=5):

    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normal_cutoff,
        btype='high'
    )

    return filtfilt(b, a, data)

# =========================================================

def apply_notch_filter(data, sr, freq=50, quality=30):

    b, a = iirnotch(
        freq,
        quality,
        sr
    )

    return filtfilt(b, a, data)

# =========================================================
# FFT + PEAK DETECTION
# =========================================================

def detect_noise_frequencies(signal, sr):

    N = len(signal)

    yf = fft(signal)
    xf = fftfreq(N, 1 / sr)

    magnitude = np.abs(yf[:N // 2])
    frequencies = xf[:N // 2]

    # =====================================================
    # PEAK DETECTION
    # =====================================================

    peaks, properties = find_peaks(
        magnitude,
        height=np.max(magnitude) * 0.2
    )

    peak_freqs = frequencies[peaks]
    peak_heights = properties["peak_heights"]

    return (
        peak_freqs,
        peak_heights,
        frequencies,
        magnitude
    )

# =========================================================
# ADAPTIVE DENOISE SYSTEM
# =========================================================

def process_and_denoise(
    noisy_signal,
    sr,
    peak_freqs,
    peak_heights
):

    filtered_signal = noisy_signal.copy()

    # =====================================================
    # LOW FREQUENCY DRIFT DETECTION
    # =====================================================

    low_freq_exists = np.any(peak_freqs < 20)

    if low_freq_exists:

        print(" -> High-pass filter uygulanıyor")

        filtered_signal = apply_highpass(
            filtered_signal,
            sr,
            cutoff=80
        )

    # =====================================================
    # ELECTRICAL NOISE DETECTION
    # =====================================================

    notch_needed = False

    for freq, amp in zip(peak_freqs, peak_heights):

        if 45 <= freq <= 55 and amp > 0.3 * np.max(peak_heights):

            notch_needed = True
            break

    if notch_needed:

        print(" -> 50 Hz Notch filter uygulanıyor")

        filtered_signal = apply_notch_filter(
            filtered_signal,
            sr,
            freq=50
        )

    # =====================================================
    # HIGH FREQUENCY CLEANUP
    # =====================================================

    print(" -> Low-pass filter uygulanıyor")

    filtered_signal = apply_lowpass(
        filtered_signal,
        sr,
        cutoff=4000
    )

    return filtered_signal

# =========================================================
# METRICS
# =========================================================

def calculate_metrics(original, filtered):

    min_length = min(
        len(original),
        len(filtered)
    )

    original = original[:min_length]
    filtered = filtered[:min_length]

    # =====================================================
    # MSE
    # =====================================================

    mse = np.mean(
        (original - filtered) ** 2
    )

    # =====================================================
    # SNR
    # =====================================================

    signal_power = np.mean(original ** 2)

    noise_power = np.mean(
        (original - filtered) ** 2
    )

    if noise_power == 0:

        snr = float('inf')

    else:

        snr = 10 * np.log10(
            signal_power / noise_power
        )

    return mse, snr

# =========================================================
# SAVE WAVEFORM
# =========================================================

def save_waveform(signal, title, path):

    plt.figure(figsize=(12, 4))

    plt.plot(signal)

    plt.title(title)

    plt.xlabel("Sample")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# =========================================================
# SAVE FFT GRAPH
# =========================================================

def save_fft_graph(
    frequencies,
    magnitude,
    peak_freqs,
    title,
    path
):

    plt.figure(figsize=(12, 4))

    plt.plot(
        frequencies,
        magnitude,
        linewidth=1
    )

    # =====================================================
    # PEAK VISUALIZATION
    # =====================================================

    peak_mask = np.isin(
        frequencies,
        peak_freqs[:10]
    )

    plt.scatter(
        frequencies[peak_mask],
        magnitude[peak_mask],
        color='red'
    )

    plt.title(title)

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Magnitude")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# =========================================================
# SAVE SPECTROGRAM
# =========================================================

def save_spectrogram(signal, sr, title, path):

    plt.figure(figsize=(12, 4))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(signal)),
        ref=np.max
    )

    librosa.display.specshow(
        D,
        sr=sr,
        x_axis='time',
        y_axis='hz'
    )

    plt.colorbar(format='%+2.0f dB')

    plt.title(title)

    plt.tight_layout()

    plt.savefig(path)

    plt.close()

# =========================================================
# MAIN
# =========================================================

def main():

    dataset_path = "dataset"

    original_path = os.path.join(
        dataset_path,
        "original"
    )

    noisy_path = os.path.join(
        dataset_path,
        "noisy"
    )

    results_path = "results"

    os.makedirs(results_path, exist_ok=True)

    # =====================================================
    # FILE LIST
    # =====================================================

    files = sorted([
        f for f in os.listdir(noisy_path)
        if f.endswith('.wav')
    ])

    print(f"\nToplam {len(files)} ses dosyası işlenecek...\n")

    metrics_report = []

    # =====================================================
    # PROCESS FILES
    # =====================================================

    for file_name in files:

        print(f"\nİşleniyor: {file_name}")

        original_file = os.path.join(
            original_path,
            file_name
        )

        noisy_file = os.path.join(
            noisy_path,
            file_name
        )

        # =================================================
        # READ AUDIO
        # =================================================

        sr_original, original_signal = wavfile.read(
            original_file
        )

        sr_noisy, noisy_signal = wavfile.read(
            noisy_file
        )

        # =================================================
        # NORMALIZATION
        # =================================================

        original_signal = (
            original_signal.astype(np.float32)
            /
            np.max(np.abs(original_signal))
        )

        noisy_signal = (
            noisy_signal.astype(np.float32)
            /
            np.max(np.abs(noisy_signal))
        )

        # =================================================
        # DETECT NOISE FREQUENCIES
        # =================================================

        (
            peak_freqs,
            peak_heights,
            frequencies,
            magnitude
        ) = detect_noise_frequencies(
            noisy_signal,
            sr_noisy
        )

        print("\nBaskın Frekanslar:")

        print(peak_freqs[:10])

        # =================================================
        # FILTERING
        # =================================================

        filtered_signal = process_and_denoise(
            noisy_signal,
            sr_noisy,
            peak_freqs,
            peak_heights
        )

        # =================================================
        # FILTERED FFT
        # =================================================

        (
            filtered_peak_freqs,
            filtered_peak_heights,
            filtered_frequencies,
            filtered_magnitude
        ) = detect_noise_frequencies(
            filtered_signal,
            sr_noisy
        )

        # =================================================
        # METRICS
        # =================================================

        mse, snr = calculate_metrics(
            original_signal,
            filtered_signal
        )

        metrics_report.append(
            (file_name, mse, snr)
        )

        # =================================================
        # SAVE FILTERED AUDIO
        # =================================================

        output_audio_path = os.path.join(
            results_path,
            f"filtered_{file_name}"
        )

        sf.write(
            output_audio_path,
            filtered_signal,
            sr_noisy
        )

        # =================================================
        # SAVE WAVEFORMS
        # =================================================

        save_waveform(
            original_signal,
            f"Original Signal - {file_name}",
            os.path.join(
                results_path,
                f"original_waveform_{file_name}.png"
            )
        )

        save_waveform(
            noisy_signal,
            f"Noisy Signal - {file_name}",
            os.path.join(
                results_path,
                f"noisy_waveform_{file_name}.png"
            )
        )

        save_waveform(
            filtered_signal,
            f"Filtered Signal - {file_name}",
            os.path.join(
                results_path,
                f"filtered_waveform_{file_name}.png"
            )
        )

        # =================================================
        # SAVE FFT
        # =================================================

        save_fft_graph(
            frequencies,
            magnitude,
            peak_freqs,
            f"Noisy FFT Spectrum - {file_name}",
            os.path.join(
                results_path,
                f"fft_noisy_{file_name}.png"
            )
        )

        save_fft_graph(
            filtered_frequencies,
            filtered_magnitude,
            filtered_peak_freqs,
            f"Filtered FFT Spectrum - {file_name}",
            os.path.join(
                results_path,
                f"fft_filtered_{file_name}.png"
            )
        )

        # =================================================
        # SAVE SPECTROGRAMS
        # =================================================

        save_spectrogram(
            noisy_signal,
            sr_noisy,
            f"Noisy Spectrogram - {file_name}",
            os.path.join(
                results_path,
                f"spectrogram_noisy_{file_name}.png"
            )
        )

        save_spectrogram(
            filtered_signal,
            sr_noisy,
            f"Filtered Spectrogram - {file_name}",
            os.path.join(
                results_path,
                f"spectrogram_filtered_{file_name}.png"
            )
        )

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 60)

    print("TÜM DOSYALAR İÇİN NİCEL ANALİZ SONUÇLARI")

    print("=" * 60)

    print(f"{'Dosya':<15} | {'MSE':<15} | {'SNR (dB)'}")

    print("-" * 60)

    for file_name, mse, snr in metrics_report:

        print(
            f"{file_name:<15} | "
            f"{mse:<15.6f} | "
            f"{snr:.2f}"
        )

    print("\nTüm sonuçlar 'results' klasörüne kaydedildi.")

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()