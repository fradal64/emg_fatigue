from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from loguru import logger
from matplotlib.ticker import MaxNLocator

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
    test_recording_indices: Optional[Dict[str, Dict[str, List[int]]]] = None,
    num_thumbnails: int = 20,
) -> None:
    """
    Visualizes model predictions against true labels for each recording in the test set.
    Applies normalization if mean and std are provided.
    If test_recording_indices is provided, only visualizes recordings with indices in that list.
    Uses a top plot for fatigue labels and a bottom row of spectrogram thumbnails.

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
        test_recording_indices: Optional dictionary mapping participant ID and side to a list of
                                original recording indices to visualize. If None, visualize all.
        num_thumbnails: Number of spectrogram thumbnails to display below the main plot.
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

            # Determine which recording indices to process for this side
            allowed_indices = None
            if (
                test_recording_indices
                and p_id in test_recording_indices
                and side in test_recording_indices[p_id]
            ):
                allowed_indices = set(test_recording_indices[p_id][side])
                logger.info(
                    f"  Visualizing specific indices for {p_id}/{side}: {sorted(list(allowed_indices))}"
                )
            else:
                logger.info(
                    f"  Visualizing all found recordings for {p_id}/{side} (no specific indices provided)."
                )

            num_processed = 0
            for rec_index, recording in enumerate(recordings):
                # --- Filter recordings based on test_recording_indices ---
                if allowed_indices is not None and rec_index not in allowed_indices:
                    # logger.debug(f"  Skipping recording {rec_index + 1} for {p_id} {side} (not in test_recording_indices).")
                    continue  # Skip this recording if it's not in the allowed list

                logger.info(
                    f"  Processing recording {rec_index + 1}/{len(recordings)} for {p_id} {side}..."
                )
                num_processed += 1

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

                # --- Plotting: Scatter Labels (Top) + 1D Frequency Profiles (Bottom) + Time Axis ---
                fig = plt.figure(figsize=(15, 6.5))  # Slightly taller for bottom axis
                # Add extra row for the time axis, adjust height ratios
                gs = fig.add_gridspec(
                    3,
                    num_thumbnails,
                    height_ratios=[
                        10,
                        10,
                        1,
                    ],  # Give labels/profiles more space, time axis less
                    width_ratios=[1] * num_thumbnails,
                )

                ax_labels = fig.add_subplot(gs[0, :])  # Top row for scatter plot
                fig.suptitle(
                    f"Participant {p_id} - {side.capitalize()} Side - Rec {rec_index + 1} - Model: {model_name}"
                )

                # --- Data Preparation for Point-in-Time Plots ---
                Sxx_db = 10 * np.log10(Sxx + np.finfo(float).eps)
                target_times = np.linspace(
                    t_spectrogram.min(), t_spectrogram.max(), num_thumbnails
                )

                plot_data_points = []  # List to store tuples: (t_val, y_true, y_pred, profile, spec_time)

                # Find closest indices and gather data
                for t_target in target_times:
                    spec_idx = np.argmin(np.abs(t_spectrogram - t_target))
                    actual_spec_time = t_spectrogram[spec_idx]
                    profile = Sxx_db[:, spec_idx]

                    # Find the closest index in t_valid (non-padded label time)
                    if len(t_valid) > 0:
                        valid_idx = np.argmin(np.abs(t_valid - actual_spec_time))
                        actual_valid_time = t_valid[valid_idx]
                        # Check if the closest valid time is reasonably close to the target spec time
                        if (
                            np.abs(actual_valid_time - actual_spec_time)
                            < (t_spectrogram[1] - t_spectrogram[0]) * 2
                        ):  # Heuristic: within 2 spectrogram time steps
                            plot_data_points.append(
                                (
                                    actual_valid_time,
                                    y_true_valid[valid_idx],
                                    y_pred_valid[valid_idx],
                                    profile,
                                    actual_spec_time,
                                )
                            )
                    else:
                        logger.warning(
                            f"Skipping target time {t_target:.2f}s as t_valid is empty."
                        )

                if not plot_data_points:
                    logger.warning(
                        f"No valid data points found for plotting for {p_id} {side} Rec {rec_index + 1}. Skipping plot."
                    )
                    plt.close(fig)
                    continue

                # Unzip the collected data
                plot_t, plot_y_true, plot_y_pred, plot_profiles, plot_spec_times = zip(
                    *plot_data_points
                )
                plot_t = np.array(plot_t)
                plot_y_true = np.array(plot_y_true)
                plot_y_pred = np.array(plot_y_pred)

                # --- Top Plot: Scatter plot of Labels at specific times ---
                scatter_df = pd.DataFrame(
                    {
                        "Time (s)": np.concatenate([plot_t, plot_t]),
                        "Fatigue Progression (%)": np.concatenate(
                            [plot_y_true, plot_y_pred]
                        ),
                        "Label Type": ["True"] * len(plot_t)
                        + ["Predicted"] * len(plot_t),
                    }
                )

                sns.scatterplot(
                    data=scatter_df,
                    x="Time (s)",
                    y="Fatigue Progression (%)",
                    hue="Label Type",
                    style="Label Type",
                    s=80,  # Increase marker size
                    ax=ax_labels,
                )
                ax_labels.set_ylabel("Fatigue Progression (%)", fontsize=10)
                ax_labels.grid(True, linestyle=":")
                ax_labels.legend(loc="upper left", fontsize=10)
                ax_labels.set_ylim(-5, 105)

                # --- Align Top Plot & FORCE remove its x-axis elements ---
                # Set x-ticks and limits to precisely match the thumbnail times
                ax_labels.set_xticks(plot_t)
                # Explicitly remove labels and ticks AFTER plotting
                ax_labels.set_xticklabels([])
                ax_labels.tick_params(
                    axis="x", labelbottom=False, bottom=False, length=0
                )
                ax_labels.set_xlabel("")  # Force remove xlabel

                if len(plot_t) > 0:
                    # Calculate limits based on thumbnail spacing for better visual centering
                    if len(plot_t) > 1:
                        time_step = plot_t[1] - plot_t[0]
                        lim_min = plot_t[0] - time_step / 2
                        lim_max = plot_t[-1] + time_step / 2
                    else:
                        time_step = 5  # Default spacing if only one point
                        lim_min = plot_t[0] - time_step / 2
                        lim_max = plot_t[0] + time_step / 2
                    ax_labels.set_xlim(lim_min, lim_max)
                else:
                    ax_labels.set_xlim(0, 1)  # Default if no points

                # --- Middle Row: 1D Frequency Profiles ---
                all_profiles = np.array(plot_profiles)
                power_min = (
                    np.percentile(all_profiles, 1) if all_profiles.size > 0 else 0
                )
                power_max = (
                    np.percentile(all_profiles, 99) if all_profiles.size > 0 else 1
                )

                thumbnail_axes = []  # Store axes for later use
                for i, (profile, spec_time) in enumerate(
                    zip(plot_profiles, plot_spec_times)
                ):
                    ax_thumb = fig.add_subplot(gs[1, i])  # Use middle row (index 1)
                    thumbnail_axes.append(ax_thumb)
                    ax_thumb.plot(profile, f)  # Power on X, Frequency on Y

                    ax_thumb.set_xlim(power_min, power_max)  # Shared power axis
                    ax_thumb.set_ylim(f.min(), f.max())
                    ax_thumb.set_title("")  # Remove time from title

                    if i == 0:
                        ax_thumb.set_ylabel("Freq (Hz)", fontsize=9)
                        ax_thumb.tick_params(axis="y", labelsize=8)
                    else:
                        ax_thumb.set_yticks([])
                        ax_thumb.set_yticklabels([])

                    # Show power ticks on all thumbnails
                    ax_thumb.xaxis.set_major_locator(
                        MaxNLocator(nbins=3, prune="both")
                    )  # Limit ticks
                    ax_thumb.tick_params(axis="x", labelsize=8)
                    ax_thumb.set_xlabel("")  # Ensure individual xlabel is cleared

                    # Set xlabel ONLY on the first plot
                    if i == 0:
                        ax_thumb.set_xlabel(
                            "Power (dB)", fontsize=9
                        )  # Add label to first plot

                    ax_thumb.grid(True, linestyle=":", alpha=0.5)

                # --- Bottom Row: Time Axis ---
                ax_time = fig.add_subplot(gs[2, :])  # Use bottom row (index 2)
                ax_time.set_xlim(
                    ax_labels.get_xlim()
                )  # Match limits with the label plot
                ax_time.set_xticks(plot_t)  # Set ticks to the exact times
                ax_time.set_xticklabels([f"{t:.1f}s" for t in plot_t], fontsize=9)
                ax_time.set_xlabel(
                    "Time (s)", fontsize=10, labelpad=15
                )  # Increase fontsize for Time label & add padding
                ax_time.set_yticks([])  # No y-ticks needed

                # Keep only bottom spine for a line effect
                ax_time.spines["top"].set_visible(False)
                ax_time.spines["left"].set_visible(False)
                ax_time.spines["right"].set_visible(False)
                ax_time.spines["bottom"].set_linewidth(1.0)  # Standard line width

                ax_time.tick_params(axis="x", length=4)

                # --- Layout Adjustments ---
                plt.tight_layout(rect=[0, 0.03, 1, 0.92])
                fig.subplots_adjust(hspace=0.6, wspace=0.2)  # Adjust hspace if needed

                # --- Save figure ---
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
            ax=ax_abs,
        )

        ax_abs.set_xlabel(
            "Time (s)",
        )
        ax_abs.set_ylabel("Fatigue Progression (%)")
        ax_abs.set_title("Overlay of True vs. Predicted Fatigue Labels")
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
                unique_labels.values(),
                unique_labels.keys(),
                fontsize=10,
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
            ax=ax_norm,
        )

        ax_norm.set_xlabel("Time (% of task duration)")
        ax_norm.set_ylabel("Fatigue Progression (%)")
        ax_norm.set_title(
            "Overlay of True vs. Predicted Fatigue Labels (Normalized Time)"
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
                fontsize=10,
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
