"""
Demonstration of spectrogram data augmentation techniques using time and frequency masking.
Based on SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from emg_fatigue.utils.augmentation import apply_freq_mask, apply_time_mask
from emg_fatigue.utils.load_emg_data import load_all_participant_data
from emg_fatigue.utils.process_emg_data import process_all_participant_data


def generate_spectrogram_grid(
    spec: np.ndarray,
    time_vec: np.ndarray,
    freq_vec: np.ndarray,
    time_params: list = [10, 20, 30],
    freq_params: list = [5, 10, 15],
):
    """
    Generate a grid of augmented spectrograms with different time and frequency masking parameters.

    Args:
        spec: Original spectrogram (freq_bins, time_bins)
        time_vec: Time vector
        freq_vec: Frequency vector
        time_params: List of time masking parameters to test
        freq_params: List of frequency masking parameters to test

    Returns:
        None (displays plot)
    """
    # First, transpose the spectrogram to match the required format for masking
    # From (freq_bins, time_bins) to (time_bins, freq_bins)
    spec_transposed = spec.T

    # Convert to dB for plotting
    def to_db(s):
        return 10 * np.log10(s + np.finfo(float).eps)

    # Create the figure
    n_rows = len(freq_params) + 1
    n_cols = len(time_params) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Plot the original spectrogram in the top-left corner
    axes[0, 0].pcolormesh(
        time_vec, freq_vec, to_db(spec), shading="gouraud", cmap="viridis"
    )
    axes[0, 0].set_title("Original Spectrogram")
    axes[0, 0].set_ylabel("Frequency (Hz)")
    axes[0, 0].set_xlabel("Time (s)")

    # Plot spectrograms with only time masking in the top row
    for i, time_param in enumerate(time_params):
        # Apply time masking
        time_masked = apply_time_mask(
            spec_transposed, time_param=time_param, num_masks=2
        )
        # Transpose back for plotting
        time_masked = time_masked.T

        # Plot
        axes[0, i + 1].pcolormesh(
            time_vec, freq_vec, to_db(time_masked), shading="gouraud", cmap="viridis"
        )
        axes[0, i + 1].set_title(f"Time Mask (param={time_param})")
        axes[0, i + 1].set_ylabel("Frequency (Hz)")
        axes[0, i + 1].set_xlabel("Time (s)")

    # Plot spectrograms with only frequency masking in the first column
    for i, freq_param in enumerate(freq_params):
        # Apply frequency masking
        freq_masked = apply_freq_mask(
            spec_transposed, freq_param=freq_param, num_masks=2
        )
        # Transpose back for plotting
        freq_masked = freq_masked.T

        # Plot
        axes[i + 1, 0].pcolormesh(
            time_vec, freq_vec, to_db(freq_masked), shading="gouraud", cmap="viridis"
        )
        axes[i + 1, 0].set_title(f"Freq Mask (param={freq_param})")
        axes[i + 1, 0].set_ylabel("Frequency (Hz)")
        axes[i + 1, 0].set_xlabel("Time (s)")

    # Plot spectrograms with both time and frequency masking
    for i, freq_param in enumerate(freq_params):
        for j, time_param in enumerate(time_params):
            # Apply both maskings (time then frequency)
            masked = apply_time_mask(
                spec_transposed, time_param=time_param, num_masks=2
            )
            masked = apply_freq_mask(masked, freq_param=freq_param, num_masks=2)
            # Transpose back for plotting
            masked = masked.T

            # Plot
            axes[i + 1, j + 1].pcolormesh(
                time_vec, freq_vec, to_db(masked), shading="gouraud", cmap="viridis"
            )
            axes[i + 1, j + 1].set_title(f"Time={time_param}, Freq={freq_param}")
            axes[i + 1, j + 1].set_ylabel("Frequency (Hz)")
            axes[i + 1, j + 1].set_xlabel("Time (s)")

    plt.tight_layout()

    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis"),
        cax=cbar_ax,
        label="Power/Frequency (dB/Hz)",
    )

    return fig


def visualize_augmentation_sequence(
    spec: np.ndarray,
    time_vec: np.ndarray,
    freq_vec: np.ndarray,
    time_param: int = 20,
    freq_param: int = 10,
    num_samples: int = 4,
):
    """
    Visualize a sequence of augmentations applied to the same spectrogram.

    Args:
        spec: Original spectrogram (freq_bins, time_bins)
        time_vec: Time vector
        freq_vec: Frequency vector
        time_param: Time masking parameter
        freq_param: Frequency masking parameter
        num_samples: Number of augmented samples to generate

    Returns:
        None (displays plot)
    """
    # Convert to the format needed for augmentation (time, freq)
    spec_transposed = spec.T

    # Convert to dB for plotting
    def to_db(s):
        return 10 * np.log10(s + np.finfo(float).eps)

    # Create the figure
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(5 * (num_samples + 1), 5))

    # Plot the original spectrogram
    axes[0].pcolormesh(
        time_vec, freq_vec, to_db(spec), shading="gouraud", cmap="viridis"
    )
    axes[0].set_title("Original Spectrogram")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_xlabel("Time (s)")

    # Create and plot augmented versions
    for i in range(num_samples):
        # Apply random masking
        masked = spec_transposed.copy()

        # Apply 1-3 time masks
        num_time_masks = random.randint(1, 3)
        masked = apply_time_mask(
            masked, time_param=time_param, num_masks=num_time_masks
        )

        # Apply 1-3 frequency masks
        num_freq_masks = random.randint(1, 3)
        masked = apply_freq_mask(
            masked, freq_param=freq_param, num_masks=num_freq_masks
        )

        # Transpose back for plotting
        masked = masked.T

        # Plot
        axes[i + 1].pcolormesh(
            time_vec, freq_vec, to_db(masked), shading="gouraud", cmap="viridis"
        )
        axes[i + 1].set_title(
            f"Augmented Sample {i + 1}\n({num_time_masks} time, {num_freq_masks} freq masks)"
        )
        axes[i + 1].set_xlabel("Time (s)")

    plt.tight_layout()

    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis"),
        cax=cbar_ax,
        label="Power/Frequency (dB/Hz)",
    )

    return fig


def main():
    """Main function to run the demo."""
    # Set up logging
    logger.info("Running spectrogram augmentation visualization demo")

    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "trimmed"
    figures_dir = project_root / "reports" / "figures" / "augmentation_demo"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # First, load the raw EMG data
    logger.info(f"Loading EMG data from {data_dir}")
    raw_data = load_all_participant_data(data_dir=data_dir)

    # Then, process the raw data to compute spectrograms
    logger.info("Processing raw data to compute spectrograms")
    processed_data = process_all_participant_data(raw_data)

    # Get the first participant
    first_participant_id = list(processed_data.keys())[0]
    logger.info(f"Using data from participant {first_participant_id}")

    # Get the first recording from left bicep
    if (
        "left" in processed_data[first_participant_id]
        and processed_data[first_participant_id]["left"]
    ):
        recording = processed_data[first_participant_id]["left"][0]

        # Extract spectrogram and related vectors
        spectrogram = recording["spectrogram"]
        time_vector = recording["time_vector"]
        freq_vector = recording["freq_vector"]

        # Generate the grid visualization
        logger.info("Generating parameter grid visualization")
        grid_fig = generate_spectrogram_grid(
            spectrogram,
            time_vector,
            freq_vector,
            time_params=[10, 20, 30],
            freq_params=[5, 10, 15],
        )
        grid_path = figures_dir / "augmentation_parameter_grid.png"
        grid_fig.savefig(grid_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved parameter grid visualization to {grid_path}")

        # Generate the sequence visualization
        logger.info("Generating augmentation sequence visualization")
        sequence_fig = visualize_augmentation_sequence(
            spectrogram,
            time_vector,
            freq_vector,
            time_param=20,
            freq_param=10,
            num_samples=4,
        )
        sequence_path = figures_dir / "augmentation_sequence.png"
        sequence_fig.savefig(sequence_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved augmentation sequence visualization to {sequence_path}")

        logger.info("Visualizations complete")
    else:
        logger.error(
            f"No 'left' recordings found for participant {first_participant_id}"
        )


if __name__ == "__main__":
    main()
