from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger

from emg_fatigue.config import FIGURES_DIR, PADDING_VALUE


def visualize_model_predictions(
    model: tf.keras.Model,
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    test_participant_ids: List[str],
    input_shape: Tuple[int, int],
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> None:
    """
    Visualizes model predictions against true labels for each recording in the test set.
    Applies normalization if mean and std are provided.

    Args:
        model: The trained Keras model.
        processed_data: Dictionary containing processed data for all participants.
                       Structure: {
                           'P001': {
                               'left': [
                                   {
                                       'raw_signal': np.ndarray,  # Raw EMG signal
                                       'raw_time': np.ndarray,    # Time vector for raw signal
                                       'spectrogram': np.ndarray, # Spectrogram (Sxx)
                                       'time_vector': np.ndarray, # Time vector for spectrogram
                                       'freq_vector': np.ndarray, # Frequency vector for spectrogram
                                       'labels': np.ndarray       # Fatigue labels (0-100)
                                   },
                                   # More recordings...
                               ],
                               'right': [...]  # Similar structure for right side
                           },
                           'P002': {...},  # More participants
                       }
        test_participant_ids: List of participant IDs included in the test set.
        input_shape: The input shape tuple (max_len, num_features) returned by dataset creation.
        norm_mean: Mean array for normalization (from training set). If None, no normalization.
        norm_std: Standard deviation array for normalization (from training set). If None, no normalization.
    """
    # Extract max_len from input_shape
    max_len = input_shape[0]

    logger.info(
        f"Starting prediction visualization for {len(test_participant_ids)} test participants."
    )
    if norm_mean is not None and norm_std is not None:
        logger.info(
            "Normalization statistics provided. Applying normalization to test data."
        )
    else:
        logger.info("No normalization statistics provided. Using raw spectrogram data.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for p_id in test_participant_ids:
        participant_dict = processed_data.get(p_id)
        if not participant_dict:
            logger.warning(f"Participant {p_id} not found in processed_data. Skipping.")
            continue

        logger.info(f"Visualizing predictions for participant {p_id}...")

        for side in ["left", "right"]:
            recordings: Optional[List[Dict[str, np.ndarray]]] = participant_dict.get(
                side
            )

            if recordings is None or not recordings:
                logger.warning(
                    f"No recordings found or side '{side}' key missing for participant {p_id}. Skipping side."
                )
                continue

            for rec_index, recording in enumerate(recordings):
                logger.info(
                    f"  Processing recording {rec_index + 1}/{len(recordings)} for {p_id} {side}..."
                )

                # Extract data for the current recording
                Sxx = recording.get("spectrogram")
                y_true = recording.get("labels")
                t_spectrogram = recording.get("time_vector")
                f = recording.get("freq_vector")
                raw_signal = recording.get("raw_signal")
                t_raw = recording.get("raw_time")

                if (
                    Sxx is None
                    or y_true is None
                    or t_spectrogram is None
                    or f is None
                    or raw_signal is None
                    or t_raw is None
                ):
                    logger.warning(
                        f"  Missing one or more required data keys for {p_id} {side} Rec {rec_index + 1}. Skipping recording."
                    )
                    continue

                # Ensure the time vector for the spectrogram matches the label length
                if len(t_spectrogram) != len(y_true):
                    logger.warning(
                        f"  Spectrogram time vector length ({len(t_spectrogram)}) does not match "
                        f"label length ({len(y_true)}) for {p_id} {side} Rec {rec_index + 1}. Skipping recording."
                    )
                    continue

                # --- Prepare data for model ---
                Sxx_transposed = Sxx.T  # Shape: (time_steps, features)

                # Normalize the transposed spectrogram if stats are provided
                if norm_mean is not None and norm_std is not None:
                    try:
                        Sxx_normalized = (Sxx_transposed - norm_mean) / norm_std
                        logger.debug(
                            f"  Applied normalization to spectrogram for {p_id} {side} Rec {rec_index + 1}"
                        )
                    except Exception as e:
                        logger.error(
                            f"  Error applying normalization for {p_id} {side} Rec {rec_index + 1}: {e}. Using unnormalized data."
                        )
                        Sxx_normalized = Sxx_transposed  # Fallback to unnormalized
                else:
                    Sxx_normalized = Sxx_transposed  # Use unnormalized data

                # Pad sequences to match expected input_shape
                # Use the (potentially) normalized data for padding
                padded_Sxx = pad_sequence(Sxx_normalized, max_len)
                padded_y_true = pad_sequence(y_true, max_len)
                padded_t_spectrogram = pad_sequence(t_spectrogram, max_len)

                input_batch = tf.expand_dims(padded_Sxx, axis=0)

                # --- Get Predictions ---
                try:
                    predictions_padded = model.predict(input_batch, verbose=0)[0]
                    if (
                        predictions_padded.ndim > 1
                        and predictions_padded.shape[-1] == 1
                    ):
                        predictions_padded = np.squeeze(predictions_padded, axis=-1)

                except Exception as e:
                    logger.error(
                        f"  Error during prediction for {p_id} {side} Rec {rec_index + 1}: {e}"
                    )
                    continue

                if predictions_padded.shape[0] != padded_y_true.shape[0]:
                    logger.error(
                        f"  Shape mismatch after prediction for {p_id} {side} Rec {rec_index + 1}. "
                        f"Pred shape: {predictions_padded.shape}, True shape: {padded_y_true.shape}. Skipping."
                    )
                    continue

                # --- Masking: Ignore padded steps ---
                mask = padded_y_true != PADDING_VALUE
                t_valid = padded_t_spectrogram[mask]
                y_true_valid = padded_y_true[mask]
                y_pred_valid = predictions_padded[mask]

                if len(t_valid) == 0:
                    logger.warning(
                        f"  No valid (non-padded) data points found for {p_id} {side} Rec {rec_index + 1} after masking. Skipping plot."
                    )
                    continue

                # --- Plotting ---
                fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
                fig.suptitle(
                    f"Participant {p_id} - {side.capitalize()} Side - Recording {rec_index + 1}"
                )

                # Determine common time range for all plots
                # Using the spectrogram time vector as the reference
                t_min = t_spectrogram.min() if len(t_spectrogram) > 0 else 0
                t_max = t_spectrogram.max() if len(t_spectrogram) > 0 else 100

                # Plot 1: Raw EMG Signal (Top)
                ax = axes[0]
                ax.plot(t_raw, raw_signal)
                ax.set_ylabel("Amplitude (mV)")
                ax.set_title("Raw EMG Signal")
                ax.grid(True)
                # Set consistent x-axis limits
                ax.set_xlim(t_min, t_max)

                # Plot 2: Spectrogram (Middle)
                ax = axes[1]
                # Calculate dB scale exactly as in visualize_partecipant_data.py
                # IMPORTANT: Use the ORIGINAL Sxx for plotting, not the normalized one
                Sxx_db = 10 * np.log10(Sxx + np.finfo(float).eps)

                if len(t_spectrogram) == Sxx.shape[1]:
                    pcm = ax.pcolormesh(
                        t_spectrogram,  # Use spectrogram time vector
                        f,
                        Sxx_db,
                        shading="gouraud",
                        cmap="viridis",
                    )
                    fig.colorbar(pcm, ax=ax, label="Power/Frequency (dB/Hz)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title("Spectrogram (Log Scale)")
                    # Set consistent x-axis limits
                    ax.set_xlim(t_min, t_max)
                else:
                    logger.warning(
                        f"  Spectrogram time vector length ({len(t_spectrogram)}) mismatch with Sxx columns ({Sxx.shape[1]}) for {p_id} {side} Rec {rec_index + 1}. Skipping spectrogram plot."
                    )
                    ax.set_title("Spectrogram (Time Mismatch)")
                    ax.set_ylabel("Frequency (Hz)")

                # Plot 3: True vs Predicted Labels (Bottom)
                ax = axes[2]
                ax.plot(
                    t_valid,
                    y_true_valid,
                    label="True Fatigue Label",
                    marker=".",
                    linestyle="-",
                    color="blue",
                )
                ax.plot(
                    t_valid,
                    y_pred_valid,
                    label="Predicted Fatigue Label",
                    marker="x",
                    linestyle="--",
                    color="red",
                )

                # Calculate mean absolute error for this recording
                mae = np.mean(np.abs(y_true_valid - y_pred_valid))
                ax.text(
                    0.97,
                    0.05,
                    f"MAE: {mae:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
                    ha="right",
                    va="bottom",
                )

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Fatigue Label")
                ax.set_title("True vs. Predicted Fatigue Labels")
                ax.legend()
                ax.grid(True)
                # Set consistent x-axis limits
                ax.set_xlim(t_min, t_max)

                # Adjust layout for better spacing
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # --- Save figure ---
                # Create a separate folder for each participant
                participant_dir = FIGURES_DIR / p_id
                participant_dir.mkdir(parents=True, exist_ok=True)

                # Save to participant-specific directory with a simpler filename
                save_path = (
                    participant_dir / f"predictions_{side}_rec{rec_index + 1}.png"
                )
                try:
                    plt.savefig(save_path)
                    logger.info(f"  Saved prediction plot to: {save_path}")
                except Exception as e:
                    logger.error(f"  Failed to save plot {save_path}: {e}")
                plt.close(fig)

    logger.info("Finished prediction visualization.")


def pad_sequence(sequence: np.ndarray, max_len: int) -> np.ndarray:
    """
    Pads or truncates a single sequence to the specified maximum length.
    Handles both 1D (labels, time) and 2D (features) arrays.
    """
    current_len = sequence.shape[0]

    if current_len == max_len:
        return sequence
    elif current_len > max_len:
        # Truncate
        return sequence[:max_len]
    else:
        # Pad
        pad_len = max_len - current_len
        # Define padding width based on dimension
        if sequence.ndim == 1:
            # Pad 1D array (e.g., labels, time)
            pad_width = (0, pad_len)
        elif sequence.ndim == 2:
            # Pad 2D array (e.g., features) - pad only the first dimension (time)
            pad_width = ((0, pad_len), (0, 0))
        else:
            raise ValueError("Padding only supported for 1D and 2D arrays.")

        return np.pad(
            sequence, pad_width, mode="constant", constant_values=PADDING_VALUE
        )
