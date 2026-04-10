import librosa

def reduce_noise(signal, sr):
    """
    Removes silent parts from audio signal
    """
    
    trimmed_signal, index = librosa.effects.trim(signal, top_db=20)

    print("\n--- SILENCE REMOVAL ---")
    print(f"Original Length: {len(signal)}")
    print(f"Trimmed Length: {len(trimmed_signal)}")

    return trimmed_signal