import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def extract_mfcc(signal, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    print("\n--- MFCC INFO ---")
    print(f"Shape: {mfcc.shape}")

    visualize_mfcc(mfcc, sr)

    return mfcc



def visualize_mfcc(mfcc, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    plt.show()