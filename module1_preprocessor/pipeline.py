from module1_preprocessor.input_handler import load_audio
from module1_preprocessor.normalization import normalize_audio
from module1_preprocessor.noise_removal import reduce_noise

from module2_preprocessor.spectrogram import generate_spectrogram


def run_pipeline(audio_path):
    # Step 1: Load
    signal, sr = load_audio(audio_path)

    # Step 2: Normalize
    signal = normalize_audio(signal)

    # Step 3: Noise Removal
    signal = reduce_noise(signal, sr)

    # Step 4: Spectrogram
    spec = generate_spectrogram(signal, sr)

    return spec


def process_audio(audio_path):
    """Backward-compatible alias used by dataset loading and inference code."""
    return run_pipeline(audio_path)