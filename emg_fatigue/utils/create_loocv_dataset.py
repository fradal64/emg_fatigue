from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from loguru import logger

from emg_fatigue.utils.augmentation import augment_data


def ensure_label_shape(y_padded: np.ndarray) -> np.ndarray:
    """
    Ensure labels have a consistent shape for the model.

    Args:
        y_padded: Padded label array

    Returns:
        Reshaped label array
    """
    if y_padded.ndim == 3 and y_padded.shape[-1] == 1:
        return np.squeeze(y_padded, axis=-1)
    # Handle case where only one sequence was passed to pad_sequences
    elif y_padded.ndim == 1 and len(y_padded) > 0:
        return np.expand_dims(y_padded, axis=0)  # Add batch dimension
    return y_padded


def create_tf_dataset(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> Optional[tf.data.Dataset]:
    """
    Create a TensorFlow dataset from input arrays.

    Args:
        X: Input features array
        y: Target labels array
        batch_size: Batch size for the dataset

    Returns:
        TensorFlow dataset or None if creation fails
    """
    if len(X) == 0:
        logger.warning("    Skipping dataset creation for empty data.")
        return None
    try:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        logger.info(f"    Dataset created successfully with {len(X)} sequences.")
        return dataset
    except Exception as e:
        logger.error(f"    Error creating dataset: {e}")
        logger.error(f"    X shape: {X.shape}, y shape: {y.shape}")
        return None


# --- Helper functions for Normalization ---


def calculate_normalization_stats(
    data_list: List[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculate mean and std dev across all time steps and sequences for normalization."""
    if not data_list:
        return None, None
    try:
        # Concatenate all spectrograms along the time-step axis (axis 0)
        # Each item in data_list is (time_steps, features)
        all_data = np.concatenate(data_list, axis=0)
        mean = np.mean(all_data, axis=0)
        std = np.std(all_data, axis=0)
        # Add a small epsilon to std dev to prevent division by zero
        std = np.where(std == 0, 1e-6, std)
        return mean, std
    except ValueError as e:
        logger.error(f"Error concatenating data for normalization: {e}")
        # Log shapes for debugging
        # for i, arr in enumerate(data_list):
        #     logger.debug(f"    Shape of array {i}: {arr.shape}")
        return None, None


def normalize_data(
    data_list: List[np.ndarray], mean: np.ndarray, std: np.ndarray
) -> List[np.ndarray]:
    """Normalize list of spectrograms using pre-calculated mean and std."""
    if mean is None or std is None:
        logger.warning("Normalization stats are None, skipping normalization.")
        return data_list
    return [(data - mean) / std for data in data_list]


# --- Main Dataset Creation Function ---


def create_loocv_dataset(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    train_participant_ids: List[str],
    validation_participant_ids: List[str],
    test_participant_ids: List[str],
    batch_size: int,
    padding_value: float,
    normalize: bool,
    augment: bool = False,
    time_param: int = 10,
    time_masks: int = 1,
    freq_param: int = 10,
    freq_masks: int = 1,
    augmentation_factor: int = 1,
) -> Tuple[
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    Optional[Tuple[int, int]],
    Optional[int],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Creates TensorFlow datasets suitable for training an RNN using spectrograms and fatigue labels.

    Applies padding to handle variable sequence lengths. Assumes leave-one-out cross-validation
    structure but allows specifying multiple validation/test participants.

    Optionally applies optional normalization based on training data and padding for variable lengths.

    Optionally applies SpecAugment (time and frequency masking) data augmentation on training data.

    Args:
        processed_data: Dictionary containing processed EMG data, including 'spectrogram' and 'labels'.
                        Structure: {'P001': {'left': [{'spectrogram': Sxx, 'labels': lbl, ...}, ...], 'right': [...]}, ...}
        train_participant_ids: List of participant IDs for the training set.
        validation_participant_ids: List of participant IDs for the validation set.
        test_participant_ids: List of participant IDs for the test set.
        batch_size: Batch size for the TensorFlow datasets.
        padding_value: Value to use for padding sequences.
        normalize: If True, apply Z-score normalization based on training data statistics.
        augment: If True, apply SpecAugment data augmentation to the training data.
        time_param: Maximum length of time mask for SpecAugment.
        time_masks: Number of time masks to apply.
        freq_param: Maximum number of frequency channels to mask for SpecAugment.
        freq_masks: Number of frequency masks to apply.
        augmentation_factor: Number of augmented copies to create for each original spectrogram.

    Returns:
        A tuple containing:
        - train_dataset: tf.data.Dataset for training (or None if no training data).
        - val_dataset: tf.data.Dataset for validation (or None if no validation data).
        - test_dataset: tf.data.Dataset for testing (or None if no test data).
        - input_shape: Tuple (max_sequence_length, num_features) for model input.
        - output_shape: Integer (max_sequence_length) for model output.
        - norm_mean: Mean calculated from training spectrograms (or None).
        - norm_std: Standard deviation calculated from training spectrograms (or None).
        Returns (None, None, None, None, None, None, None) if no data is processed.
    """
    specs_by_set = {"train": [], "val": [], "test": []}
    labels_by_set = {"train": [], "val": [], "test": []}
    max_len = 0
    num_features = None
    norm_mean = None
    norm_std = None

    logger.info("Processing data for datasets...")

    # --- Collect data from specified participants ---
    for set_name, participant_ids in [
        ("train", train_participant_ids),
        ("val", validation_participant_ids),
        ("test", test_participant_ids),
    ]:
        logger.info(
            f"  Processing {set_name} set ({len(participant_ids)} participants)..."
        )
        for p_id in participant_ids:
            if p_id not in processed_data:
                logger.warning(
                    f"    Warning: Participant {p_id} not found in processed_data. Skipping."
                )
                continue
            for side in ["left", "right"]:
                if side not in processed_data[p_id]:
                    # logger.debug(f"    Debug: Side {side} not found for participant {p_id}.") # Optional debug
                    continue
                if not processed_data[p_id][side]:
                    # logger.debug(f"    Debug: No recordings for {p_id} side {side}.") # Optional debug
                    continue

                for recording_idx, recording in enumerate(processed_data[p_id][side]):
                    # Spectrogram Sxx: shape (freq_bins, time_steps)
                    spectrogram = recording.get("spectrogram")
                    # Labels: shape (time_steps,)
                    labels = recording.get("labels")

                    if spectrogram is None or labels is None:
                        logger.warning(
                            f"    Warning: Missing 'spectrogram' or 'labels' for {p_id}/{side}/Rec{recording_idx}. Skipping."
                        )
                        continue

                    # Validate shapes: Spectrogram time dimension should match labels length
                    if spectrogram.shape[1] != len(labels):
                        logger.warning(
                            f"    Warning: Shape mismatch for {p_id}/{side}/Rec{recording_idx}. "
                            f"Spectrogram time bins ({spectrogram.shape[1]}) != Label length ({len(labels)}). Skipping."
                        )
                        continue

                    # Transpose spectrogram: (freq_bins, time_steps) -> (time_steps, freq_bins)
                    transposed_spec = spectrogram.T
                    current_len = transposed_spec.shape[0]
                    current_feats = transposed_spec.shape[1]

                    # Determine num_features from the first valid recording
                    if num_features is None:
                        num_features = current_feats
                    # Ensure consistency
                    elif num_features != current_feats:
                        raise ValueError(
                            f"Inconsistent number of frequency bins found! "
                            f"Expected {num_features}, got {current_feats} for {p_id}/{side}/Rec{recording_idx}."
                        )

                    max_len = max(
                        max_len, current_len
                    )  # Update max length across all data

                    specs_by_set[set_name].append(transposed_spec)
                    labels_by_set[set_name].append(labels)

    if num_features is None or max_len == 0:
        logger.error("Error: No valid data found to create datasets.")
        return None, None, None, None, None, None, None

    logger.info(f"\nFound {num_features} frequency bins (features).")
    logger.info(f"Maximum sequence length across all sets: {max_len}.")

    # --- Normalization (Optional) ---
    if normalize:
        logger.info("Calculating normalization statistics from training data...")
        norm_mean, norm_std = calculate_normalization_stats(specs_by_set["train"])
        if norm_mean is not None and norm_std is not None:
            logger.info(
                f"  Calculated Mean shape: {norm_mean.shape}, Std shape: {norm_std.shape}"
            )
            # logger.debug(f"  Mean values (sample): {norm_mean[:5]}") # Optional debug
            # logger.debug(f"  Std values (sample): {norm_std[:5]}")   # Optional debug

            logger.info("Applying normalization to all datasets...")
            specs_by_set["train"] = normalize_data(
                specs_by_set["train"], norm_mean, norm_std
            )
            specs_by_set["val"] = normalize_data(
                specs_by_set["val"], norm_mean, norm_std
            )
            specs_by_set["test"] = normalize_data(
                specs_by_set["test"], norm_mean, norm_std
            )
            logger.info("Normalization complete.")
        else:
            logger.error(
                "Failed to calculate normalization statistics. Skipping normalization."
            )
            norm_mean = None
            norm_std = None

    # --- Data Augmentation (Optional) ---
    if augment and specs_by_set["train"]:
        original_count = len(specs_by_set["train"])
        original_labels = labels_by_set["train"]

        logger.info("Applying SpecAugment data augmentation to training data...")
        logger.info(f"  Time masking: param={time_param}, masks={time_masks}")
        logger.info(f"  Frequency masking: param={freq_param}, masks={freq_masks}")
        logger.info(f"  Augmentation factor: {augmentation_factor}")

        # Augment spectrograms
        augmented_specs = augment_data(
            specs_by_set["train"],
            time_param=time_param,
            time_masks=time_masks,
            freq_param=freq_param,
            freq_masks=freq_masks,
            augmentation_factor=augmentation_factor,
        )

        # Duplicate labels for augmented data
        augmented_labels = list(original_labels)
        for _ in range(augmentation_factor):
            augmented_labels.extend(original_labels)

        # Update training data
        specs_by_set["train"] = augmented_specs
        labels_by_set["train"] = augmented_labels

        logger.info(
            f"  Data augmentation complete. Training samples: {original_count} → {len(specs_by_set['train'])}"
        )

    # --- Padding Sequences ---
    logger.info("Padding sequences...")
    padded_X = {}
    padded_y = {}

    for dataset_type in ["train", "val", "test"]:
        # Pad spectrogram sequences
        padded_X[dataset_type] = (
            tf.keras.preprocessing.sequence.pad_sequences(
                specs_by_set[dataset_type],
                maxlen=max_len,
                padding="post",
                dtype="float32",
                value=padding_value,  # Use padding value for features
                truncating="post",
            )
            if specs_by_set[dataset_type]
            else np.array([])
        )

        # Pad label sequences
        padded_y[dataset_type] = (
            tf.keras.preprocessing.sequence.pad_sequences(
                labels_by_set[dataset_type],
                maxlen=max_len,
                padding="post",
                dtype="float32",
                value=padding_value,
                truncating="post",
            )
            if labels_by_set[dataset_type]
            else np.array([])
        )

        # Print shape information
        logger.info(
            f"  Padded {dataset_type.capitalize()} data shape: X={padded_X[dataset_type].shape}, y={padded_y[dataset_type].shape}"
        )

    # Rename variables for clarity in the rest of the code
    X_train_padded, y_train_padded = padded_X["train"], padded_y["train"]
    X_val_padded, y_val_padded = padded_X["val"], padded_y["val"]
    X_test_padded, y_test_padded = padded_X["test"], padded_y["test"]

    # Apply shape standardization
    y_train_padded = ensure_label_shape(y_train_padded)
    y_val_padded = ensure_label_shape(y_val_padded)
    y_test_padded = ensure_label_shape(y_test_padded)

    logger.info(
        f"  Standardized Training data shape: X={X_train_padded.shape}, y={y_train_padded.shape}"
    )
    logger.info(
        f"  Standardized Validation data shape: X={X_val_padded.shape}, y={y_val_padded.shape}"
    )
    logger.info(
        f"  Standardized Test data shape: X={X_test_padded.shape}, y={y_test_padded.shape}"
    )

    # --- Create TensorFlow Datasets ---
    logger.info("Creating TensorFlow Datasets...")

    train_dataset = create_tf_dataset(X_train_padded, y_train_padded, batch_size)
    val_dataset = create_tf_dataset(X_val_padded, y_val_padded, batch_size)
    test_dataset = create_tf_dataset(X_test_padded, y_test_padded, batch_size)

    # --- Determine Shapes ---
    # Input shape for the model is (time_steps, features)
    input_shape = (max_len, num_features)
    # Output shape is the sequence length for labels
    output_shape = max_len

    logger.info("\nDataset creation complete.")
    logger.info(f"  Input Shape for Model: {input_shape}")
    logger.info(f"  Output Shape (Sequence Length): {output_shape}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        input_shape,
        output_shape,
        norm_mean,
        norm_std,
    )
