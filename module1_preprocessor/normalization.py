import numpy as np

def normalize_audio(signal):
    """
    Normalize audio signal to range [-1, 1]
    """
    
    max_val = np.max(np.abs(signal))

    # Avoid division by zero
    if max_val == 0:
        return signal

    normalized_signal = signal / max_val

    print("\n--- NORMALIZATION ---")
    print(f"Max value before: {max_val}")
    print(f"Max value after: {np.max(np.abs(normalized_signal))}")

    return normalized_signal