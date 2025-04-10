from typing import Dict, List

import numpy as np
import pandas as pd

from emg_fatigue.config import NOVERLAP, NPERSEG, SAMPLING_FREQUENCY
from emg_fatigue.utils.compute_spectrogram import compute_spectrogram


def process_all_participant_data(
    participant_data: Dict[str, Dict[str, List[pd.DataFrame]]],
) -> Dict[str, Dict[str, List[Dict[str, np.ndarray]]]]:
    """
    Processes raw EMG data for all participants to compute spectrograms and fatigue labels,
    while keeping the raw signal.

    Args:
        participant_data: Dictionary containing raw EMG data loaded by load_all_participant_data.
                          Structure: {'P001': {'left': [df1, df2], 'right': [df3, df4]}, ...}
        nperseg: Length of each segment for spectrogram calculation.
        noverlap: Number of points to overlap between segments for spectrogram calculation.

    Returns:
        Dictionary with processed data. Structure mirrors the input, but DataFrames
        are replaced by dictionaries containing:
        {
            'raw_signal': np.ndarray,    # Raw sEMG signal
            'spectrogram': np.ndarray,  # Spectrogram (Sxx)
            'time_vector': np.ndarray,  # Time vector for the spectrogram (t)
            'freq_vector': np.ndarray,  # Frequency vector for the spectrogram (f)
            'labels': np.ndarray       # Linearly increasing fatigue labels (0-100)
        }
    """
    processed_data = {}
    for participant_id, sides_data in participant_data.items():
        processed_data[participant_id] = {"left": [], "right": []}
        for side, recordings in sides_data.items():
            for df in recordings:
                emg_signal = df["sEMG"].values
                time_signal = df[
                    "time_sEMG_seconds"
                ].values  # Also keep the time signal
                # Compute spectrogram
                f, t, Sxx = compute_spectrogram(
                    signal_data=emg_signal,
                    fs=SAMPLING_FREQUENCY,
                    nperseg=NPERSEG,
                    noverlap=NOVERLAP,
                )

                # Generate linear labels (0 to 100) corresponding to spectrogram time points
                labels = np.linspace(0, 100, len(t))

                processed_data[participant_id][side].append(
                    {
                        "raw_signal": emg_signal,
                        "raw_time": time_signal,
                        "spectrogram": Sxx,
                        "time_vector": t,
                        "freq_vector": f,
                        "labels": labels,
                    }
                )
    return processed_data
