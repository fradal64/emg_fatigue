from typing import Tuple

import numpy as np
from scipy import signal


def compute_spectrogram(
    signal_data: np.ndarray,
    nperseg: int,
    noverlap: int,
    fs: float,  # Sampling frequency in Hz
    fmax: float = 500.0,  # Maximum frequency to retain
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Spectrogram of a signal using STFT and truncate frequencies above fmax.

    Args:
        signal_data: Array containing the signal data.
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment.
        noverlap: Number of points to overlap between segments.
        fmax: Maximum frequency to retain in the output (Hz).

    Returns:
        f_truncated: Array of sample frequencies (up to fmax).
        t: Array of segment times.
        Sxx_truncated: Spectrogram of the signal (only frequencies <= fmax).
    """
    # Compute the Short-Time Fourier Transform (STFT)
    f, t, Zxx = signal.stft(
        signal_data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Calculate the power spectral density (PSD)
    Sxx = np.abs(Zxx) ** 2

    # Truncate frequencies above fmax
    freq_mask = f <= fmax
    f_truncated = f[freq_mask]
    Sxx_truncated = Sxx[freq_mask, :]

    return f_truncated, t, Sxx_truncated
