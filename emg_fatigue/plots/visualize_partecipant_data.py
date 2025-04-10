from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def print_participant_data_shapes(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    participant_id: str,
):
    """Prints the shapes of all data arrays for a given participant."""
    if participant_id not in processed_data:
        print(f"Error: Participant ID {participant_id} not found in processed data.")
        return

    participant_recordings = processed_data[participant_id]
    left_recordings = participant_recordings.get("left", [])
    right_recordings = participant_recordings.get("right", [])

    print(f"\n--- Data Shapes for Participant {participant_id} ---")

    if not left_recordings and not right_recordings:
        print("No recordings found.")
        return

    if left_recordings:
        print("\nLeft Bicep:")
        for i, data in enumerate(left_recordings):
            print(f"  Recording {i + 1}:")
            print(f"    raw_signal: {data['raw_signal'].shape}")
            print(f"    raw_time: {data['raw_time'].shape}")
            print(f"    spectrogram: {data['spectrogram'].shape}")
            print(f"    time_vector: {data['time_vector'].shape}")
            print(f"    freq_vector: {data['freq_vector'].shape}")
            print(f"    labels: {data['labels'].shape}")

    if right_recordings:
        print("\nRight Bicep:")
        for i, data in enumerate(right_recordings):
            print(f"  Recording {i + 1}:")
            print(f"    raw_signal: {data['raw_signal'].shape}")
            print(f"    raw_time: {data['raw_time'].shape}")
            print(f"    spectrogram: {data['spectrogram'].shape}")
            print(f"    time_vector: {data['time_vector'].shape}")
            print(f"    freq_vector: {data['freq_vector'].shape}")
            print(f"    labels: {data['labels'].shape}")
    print("\n---------------------------------------------")


def plot_raw_signals(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    participant_id: str,
):
    """Visualizes the raw EMG signals for a given participant (left and right)."""
    if participant_id not in processed_data:
        print(f"Error: Participant ID {participant_id} not found in processed data.")
        return

    participant_recordings = processed_data[participant_id]
    left_recordings = participant_recordings.get("left", [])
    right_recordings = participant_recordings.get("right", [])

    n_left = len(left_recordings)
    n_right = len(right_recordings)
    max_recordings = max(n_left, n_right)

    if max_recordings == 0:
        print(f"No recordings found for participant {participant_id} to plot.")
        return

    fig, axes = plt.subplots(
        max_recordings, 2, figsize=(15, 5 * max_recordings), squeeze=False
    )
    fig.suptitle(f"Participant {participant_id}: Raw EMG Signals", fontsize=16, y=1.02)

    # Set column titles
    if n_left > 0:
        axes[0, 0].set_title("Left Bicep Raw Signal", fontsize=14, pad=20)
    if n_right > 0:
        axes[0, 1].set_title("Right Bicep Raw Signal", fontsize=14, pad=20)

    for i in range(max_recordings):
        # Plot Left Raw Signal
        if i < n_left:
            data = left_recordings[i]
            axes[i, 0].plot(data["raw_time"], data["raw_signal"])
            axes[i, 0].set_ylabel(f"Rec {i + 1} Ampl. (mV)")
            axes[i, 0].grid(True)
            if i == max_recordings - 1:  # Only add x-label to bottom plot
                axes[i, 0].set_xlabel("Time (s)")
        else:
            axes[i, 0].axis("off")

        # Plot Right Raw Signal
        if i < n_right:
            data = right_recordings[i]
            axes[i, 1].plot(data["raw_time"], data["raw_signal"])
            axes[i, 1].set_ylabel(f"Rec {i + 1} Ampl. (mV)")
            axes[i, 1].grid(True)
            if i == max_recordings - 1:  # Only add x-label to bottom plot
                axes[i, 1].set_xlabel("Time (s)")
        else:
            axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_spectrograms_and_labels(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    participant_id: str,
):
    """Visualizes spectrograms and fatigue labels for a given participant."""
    if participant_id not in processed_data:
        print(f"Error: Participant ID {participant_id} not found in processed data.")
        return

    participant_recordings = processed_data[participant_id]
    left_recordings = participant_recordings.get("left", [])
    right_recordings = participant_recordings.get("right", [])

    n_left = len(left_recordings)
    n_right = len(right_recordings)
    max_recordings = max(n_left, n_right)

    if max_recordings == 0:
        print(f"No recordings found for participant {participant_id} to plot.")
        return

    fig, axes = plt.subplots(
        max_recordings, 4, figsize=(22, 5 * max_recordings), squeeze=False
    )
    fig.suptitle(
        f"Participant {participant_id}: Spectrograms and Fatigue Labels",
        fontsize=16,
        y=1.02,
    )

    # Set column titles
    if n_left > 0:
        axes[0, 0].set_title("Left Spectrogram", fontsize=14, pad=20)
        axes[0, 1].set_title("Left Labels", fontsize=14, pad=20)
    if n_right > 0:
        axes[0, 2].set_title("Right Spectrogram", fontsize=14, pad=20)
        axes[0, 3].set_title("Right Labels", fontsize=14, pad=20)

    for i in range(max_recordings):
        # Left Side Plots
        if i < n_left:
            data = left_recordings[i]
            # Spectrogram
            Sxx_db = 10 * np.log10(data["spectrogram"] + np.finfo(float).eps)
            im = axes[i, 0].pcolormesh(
                data["time_vector"],
                data["freq_vector"],
                Sxx_db,
                shading="gouraud",
                cmap="viridis",
            )
            axes[i, 0].set_ylabel(f"Rec {i + 1} Freq (Hz)")
            fig.colorbar(im, ax=axes[i, 0], label="Power/Freq (dB/Hz)")
            if i == max_recordings - 1:
                axes[i, 0].set_xlabel("Time (s)")

            # Labels
            axes[i, 1].plot(data["time_vector"], data["labels"])
            axes[i, 1].set_ylabel(f"Rec {i + 1} Fatigue Lvl")
            axes[i, 1].set_ylim(-5, 105)
            axes[i, 1].grid(True)
            if i == max_recordings - 1:
                axes[i, 1].set_xlabel("Time (s)")
        else:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

        # Right Side Plots
        if i < n_right:
            data = right_recordings[i]
            # Spectrogram
            Sxx_db = 10 * np.log10(data["spectrogram"] + np.finfo(float).eps)
            im = axes[i, 2].pcolormesh(
                data["time_vector"],
                data["freq_vector"],
                Sxx_db,
                shading="gouraud",
                cmap="viridis",
            )
            axes[i, 2].set_ylabel(f"Rec {i + 1} Freq (Hz)")
            fig.colorbar(im, ax=axes[i, 2], label="Power/Freq (dB/Hz)")
            if i == max_recordings - 1:
                axes[i, 2].set_xlabel("Time (s)")

            # Labels
            axes[i, 3].plot(data["time_vector"], data["labels"])
            axes[i, 3].set_ylabel(f"Rec {i + 1} Fatigue Lvl")
            axes[i, 3].set_ylim(-5, 105)
            axes[i, 3].grid(True)
            if i == max_recordings - 1:
                axes[i, 3].set_xlabel("Time (s)")
        else:
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


# --- Example Usage ---
# Assuming you have your 'processed_data' dictionary loaded
# from emg_fatigue.utils.load_emg_data import load_all_participant_data
# from emg_fatigue.utils.process_emg_data import process_all_participant_data
# from emg_fatigue.config import TRIMMED_DATA_DIR # Or your data directory

# # 1. Load raw data
# raw_data = load_all_participant_data(data_dir=TRIMMED_DATA_DIR)

# # 2. Process data
# processed_data = process_all_participant_data(raw_data)

# # 3. Visualize a specific participant
# participant_to_show = 'P001' # Replace with the participant ID you want to see
# if processed_data:
#    print_participant_data_shapes(processed_data, participant_to_show)
#    plot_raw_signals(processed_data, participant_to_show)
#    plot_spectrograms_and_labels(processed_data, participant_to_show)
