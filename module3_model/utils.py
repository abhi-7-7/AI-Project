import torch
import numpy as np

def to_tensor(spectrogram):
    # Normalize values
    spec = (spectrogram - np.mean(spectrogram)) / np.std(spectrogram)

    # Add channel dimension → (1, 128, 128)
    spec = np.expand_dims(spec, axis=0)

    return torch.tensor(spec, dtype=torch.float32)