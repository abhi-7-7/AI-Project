import numpy as np
import librosa

def generate_spectrogram(signal, sr, n_mels=128, max_len=128):
    spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=n_mels
    )

    log_spectrogram = librosa.power_to_db(spectrogram)

    if log_spectrogram.shape[1] < max_len:
        pad_width = max_len - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, ((0,0),(0,pad_width)))
    else:
        log_spectrogram = log_spectrogram[:, :max_len]

    print("\n--- SPECTROGRAM ---")
    print(f"Final Shape: {log_spectrogram.shape}")

    return log_spectrogram