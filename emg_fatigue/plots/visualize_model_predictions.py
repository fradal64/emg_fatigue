from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from loguru import logger

from emg_fatigue.config import FIGURES_DIR, PADDING_VALUE

# Set seaborn theme
sns.set_theme(style="whitegrid")


def visualize_model_predictions(
    model: tf.keras.Model,
    model_name: str,
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
        model_name: Name of the model to use in output filenames.
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
        f"Starting prediction visualization for {len(test_participant_ids)} test participants using model '{model_name}'."
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

        # Create a separate folder for each participant's figures
        participant_dir = FIGURES_DIR / p_id
        participant_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lists to collect data for overlay plots for the entire participant
        all_t_valid_participant = []
        all_y_true_valid_participant = []
        all_y_pred_valid_participant = []

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

                # --- Store data for participant-level overlay plots, tagged with side ---
                all_t_valid_participant.append((side, t_valid))
                all_y_true_valid_participant.append((side, y_true_valid))
                all_y_pred_valid_participant.append((side, y_pred_valid))

                # --- Plotting Individual Recording with Seaborn ---
                fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
                fig.suptitle(
                    f"Participant {p_id} - {side.capitalize()} Side - Recording {rec_index + 1} - Model: {model_name}"
                )

                # Determine common time range for all plots
                # Using the spectrogram time vector as the reference
                t_min = t_spectrogram.min() if len(t_spectrogram) > 0 else 0
                t_max = t_spectrogram.max() if len(t_spectrogram) > 0 else 100

                # Plot 1: Raw EMG Signal (Top) - Use Seaborn
                ax = axes[0]
                sns.lineplot(x=t_raw, y=raw_signal, ax=ax)
                ax.set_ylabel("Amplitude (mV)")
                ax.set_title("Raw EMG Signal")
                ax.grid(True)
                ax.set_xlim(t_min, t_max)

                # Plot 2: Spectrogram (Middle) - Keep pcolormesh
                ax = axes[1]
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

                # Plot 3: True vs Predicted Labels (Bottom) - Use Seaborn
                ax = axes[2]
                # Prepare data for seaborn lineplot (optional but good practice)
                plot_data = pd.DataFrame(
                    {
                        "time": np.concatenate([t_valid, t_valid]),
                        "fatigue": np.concatenate([y_true_valid, y_pred_valid]),
                        "type": ["True"] * len(t_valid) + ["Predicted"] * len(t_valid),
                    }
                )
                sns.lineplot(
                    data=plot_data,
                    x="time",
                    y="fatigue",
                    hue="type",
                    style="type",
                    markers=True,
                    dashes=False,
                    ax=ax,
                    palette=["blue", "red"],
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
                ax.set_xlim(t_min, t_max)

                # Adjust layout for better spacing
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                # --- Save figure ---
                # Save to participant-specific directory with model name in the filename
                save_path = (
                    participant_dir / f"{model_name}_{side}_rec{rec_index + 1}.png"
                )
                try:
                    plt.savefig(save_path)
                    logger.info(f"  Saved prediction plot to: {save_path}")
                except Exception as e:
                    logger.error(f"  Failed to save plot {save_path}: {e}")
                plt.close(fig)

        # --- Generate and save overlay plots for the PARTICIPANT after processing BOTH sides ---
        if all_t_valid_participant:  # Check if there's any valid data collected
            try:
                plot_overlay_predictions(
                    all_t_valid_participant,
                    all_y_true_valid_participant,
                    all_y_pred_valid_participant,
                    p_id,  # Pass participant ID
                    model_name,
                    participant_dir,  # Pass the participant-specific directory
                )
            except Exception as e:
                logger.error(f"Failed to generate overlay plots for {p_id}: {e}")
        else:
            logger.warning(
                f"No valid data collected for {p_id} to generate overlay plots."
            )

    logger.info(f"Finished prediction visualization for model '{model_name}'.")


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


# --- Updated Helper Function for Combined Overlay Plots using Seaborn ---
def plot_overlay_predictions(
    all_t_data: List[Tuple[str, np.ndarray]],
    all_y_true_data: List[Tuple[str, np.ndarray]],
    all_y_pred_data: List[Tuple[str, np.ndarray]],
    p_id: str,
    model_name: str,
    participant_dir: Path,
):
    """
    Generates combined overlay plots for true vs predicted fatigue labels across all
    recordings (both sides) for a participant using Seaborn.
    Creates two plots: one with absolute time and one with normalized time (0-100%).
    """
    if not all_t_data:
        logger.warning(
            f"No valid data provided for overlay plots for {p_id}. Skipping."
        )
        return

    logger.info(f"Generating combined overlay plots for participant {p_id}...")

    # --- Calculate Overall MAE (across all recordings, both sides) ---
    # Extract only the label arrays for concatenation
    y_true_combined = np.concatenate([arr for _, arr in all_y_true_data])
    y_pred_combined = np.concatenate([arr for _, arr in all_y_pred_data])
    if len(y_true_combined) > 0:
        overall_mae = np.mean(np.abs(y_true_combined - y_pred_combined))
        mae_text = f"Overall MAE: {overall_mae:.2f}"
    else:
        overall_mae = np.nan
        mae_text = "Overall MAE: N/A"

    # --- Prepare DataFrames for Seaborn --- #
    abs_plot_data_list = []
    norm_plot_data_list = []

    for i, ((side_t, t_valid), (_, y_true_valid), (_, y_pred_valid)) in enumerate(
        zip(all_t_data, all_y_true_data, all_y_pred_data)
    ):
        rec_id = f"{side_t}_{i}"
        # Absolute time data
        abs_plot_data_list.append(
            pd.DataFrame(
                {
                    "time": t_valid,
                    "fatigue": y_true_valid,
                    "type": "True",
                    "recording_id": rec_id,
                }
            )
        )
        abs_plot_data_list.append(
            pd.DataFrame(
                {
                    "time": t_valid,
                    "fatigue": y_pred_valid,
                    "type": "Predicted",
                    "recording_id": rec_id,
                }
            )
        )

        # Normalized time data
        t_min, t_max = t_valid.min(), t_valid.max()
        t_range = t_max - t_min
        if t_range > np.finfo(float).eps:
            t_norm = (t_valid - t_min) / t_range * 100.0
        elif len(t_valid) > 0:
            t_norm = np.zeros_like(t_valid) + 50.0
        else:
            t_norm = t_valid

        norm_plot_data_list.append(
            pd.DataFrame(
                {
                    "normalized_time": t_norm,
                    "fatigue": y_true_valid,
                    "type": "True",
                    "recording_id": rec_id,
                }
            )
        )
        norm_plot_data_list.append(
            pd.DataFrame(
                {
                    "normalized_time": t_norm,
                    "fatigue": y_pred_valid,
                    "type": "Predicted",
                    "recording_id": rec_id,
                }
            )
        )

    df_abs = (
        pd.concat(abs_plot_data_list, ignore_index=True)
        if abs_plot_data_list
        else pd.DataFrame()
    )
    df_norm = (
        pd.concat(norm_plot_data_list, ignore_index=True)
        if norm_plot_data_list
        else pd.DataFrame()
    )

    # --- Plot 1: Absolute Time Overlay (Seaborn) ---
    if not df_abs.empty:
        fig_abs, ax_abs = plt.subplots(1, 1, figsize=(12, 6))
        fig_abs.suptitle(
            f"Participant {p_id} - All Recordings Overlay (Absolute Time) - Model: {model_name}"
        )

        sns.lineplot(
            data=df_abs,
            x="time",
            y="fatigue",
            hue="type",  # Color by True/Predicted
            units="recording_id",  # Draw separate lines per recording
            estimator=None,  # Plot actual lines, not aggregates
            alpha=0.6,
            palette=["blue", "red"],  # Consistent colors
            ax=ax_abs,
        )

        ax_abs.set_xlabel("Time (s)")
        ax_abs.set_ylabel("Fatigue Label")
        ax_abs.set_title("Overlay of True vs. Predicted Fatigue Labels")
        ax_abs.text(
            0.97,
            0.05,
            mae_text,
            transform=ax_abs.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            ha="right",
            va="bottom",
        )
        ax_abs.grid(True)
        # Improve legend if needed (Seaborn usually does a good job)
        handles, labels = ax_abs.get_legend_handles_labels()
        # Filter unique labels (Seaborn might duplicate based on units)
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels and label in ["True", "Predicted"]:
                unique_labels[label] = handle
        if unique_labels:
            ax_abs.legend(
                unique_labels.values(), unique_labels.keys(), title="Label Type"
            )
        else:
            ax_abs.legend().set_visible(False)  # Hide legend if empty

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path_abs = participant_dir / f"{model_name}_overlay_absolute_time.png"
        try:
            plt.savefig(save_path_abs)
            logger.info(f"  Saved absolute time overlay plot to: {save_path_abs}")
        except Exception as e:
            logger.error(f"  Failed to save plot {save_path_abs}: {e}")
        plt.close(fig_abs)
    else:
        logger.warning(f"No data for absolute time overlay plot for {p_id}.")

    # --- Plot 2: Normalized Time Overlay (Seaborn) ---
    if not df_norm.empty:
        fig_norm, ax_norm = plt.subplots(1, 1, figsize=(12, 6))
        fig_norm.suptitle(
            f"Participant {p_id} - All Recordings Overlay (Normalized Time) - Model: {model_name}"
        )

        sns.lineplot(
            data=df_norm,
            x="normalized_time",
            y="fatigue",
            hue="type",
            units="recording_id",
            estimator=None,
            alpha=0.6,
            palette=["blue", "red"],
            ax=ax_norm,
        )

        ax_norm.set_xlabel("Normalized Recording Time (%)")
        ax_norm.set_ylabel("Fatigue Label")
        ax_norm.set_title(
            "Overlay of True vs. Predicted Fatigue Labels (Normalized Time)"
        )
        ax_norm.text(
            0.97,
            0.05,
            mae_text,
            transform=ax_norm.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            ha="right",
            va="bottom",
        )
        ax_norm.grid(True)
        ax_norm.set_xlim(0, 100)

        # Improve legend (similar to absolute plot)
        handles_norm, labels_norm = ax_norm.get_legend_handles_labels()
        unique_labels_norm = {}
        for handle, label in zip(handles_norm, labels_norm):
            if label not in unique_labels_norm and label in ["True", "Predicted"]:
                unique_labels_norm[label] = handle
        if unique_labels_norm:
            ax_norm.legend(
                unique_labels_norm.values(),
                unique_labels_norm.keys(),
                title="Label Type",
            )
        else:
            ax_norm.legend().set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path_norm = participant_dir / f"{model_name}_overlay_normalized_time.png"
        try:
            plt.savefig(save_path_norm)
            logger.info(f"  Saved normalized time overlay plot to: {save_path_norm}")
        except Exception as e:
            logger.error(f"  Failed to save plot {save_path_norm}: {e}")
        plt.close(fig_norm)
    else:
        logger.warning(f"No data for normalized time overlay plot for {p_id}.")
