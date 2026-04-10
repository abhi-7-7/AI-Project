import librosa

def load_audio(file_path, sample_rate=22050):
    # Step 1: Load and convert audio into numerical signal
    signal, sr = librosa.load(file_path, sr=sample_rate)

    # Step 2: Understand the signal
    duration = len(signal) / sr  # total time in seconds

    print("\n--- AUDIO INFO ---")
    print(f"Sample Rate (sr): {sr} samples/sec")
    print(f"Total Samples: {len(signal)}")
    print(f"Duration: {duration:.2f} seconds")

    # Step 3: Inspect first few values
    print("\nFirst 10 signal values:")
    print(signal[:10])

    return signal, sr