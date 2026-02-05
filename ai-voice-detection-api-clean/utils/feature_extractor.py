import librosa
import numpy as np


def extract_voice_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    if len(y) == 0:
        raise ValueError("Empty audio file")

    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch = pitch[np.isfinite(pitch)]

    pitch_mean = float(np.mean(pitch)) if len(pitch) else 0
    pitch_std = float(np.std(pitch)) if len(pitch) else 0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = float(np.mean(mfcc))
    mfcc_std = float(np.std(mfcc))

    energy = float(np.mean(librosa.feature.rms(y=y)))

    return {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "energy": energy
    }
