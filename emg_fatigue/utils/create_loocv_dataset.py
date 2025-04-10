from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf


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
        print("    Skipping dataset creation for empty data.")
        return None
    try:
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print(f"    Dataset created successfully with {len(X)} sequences.")
        return dataset
    except Exception as e:
        print(f"    Error creating dataset: {e}")
        print(f"    X shape: {X.shape}, y shape: {y.shape}")
        return None


def create_loocv_dataset(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
    train_participant_ids: List[str],
    validation_participant_ids: List[str],
    test_participant_ids: List[str],
    batch_size: int,
    padding_value: float,
) -> Tuple[
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    Optional[tf.data.Dataset],
    Optional[Tuple[int, int]],
    Optional[int],
]:
    """
    Creates TensorFlow datasets suitable for training an RNN using spectrograms and fatigue labels.

    Applies padding to handle variable sequence lengths. Assumes leave-one-out cross-validation
    structure but allows specifying multiple validation/test participants.

    Args:
        processed_data: Dictionary containing processed EMG data, including 'spectrogram' and 'labels'.
                        Structure: {'P001': {'left': [{'spectrogram': Sxx, 'labels': lbl, ...}, ...], 'right': [...]}, ...}
        train_participant_ids: List of participant IDs for the training set.
        validation_participant_ids: List of participant IDs for the validation set.
        test_participant_ids: List of participant IDs for the test set.
        batch_size: Batch size for the TensorFlow datasets.
        padding_value: Value to use for padding sequences.

    Returns:
        A tuple containing:
        - train_dataset: tf.data.Dataset for training (or None if no training data).
        - val_dataset: tf.data.Dataset for validation (or None if no validation data).
        - test_dataset: tf.data.Dataset for testing (or None if no test data).
        - input_shape: Tuple (max_sequence_length, num_features) for model input.
        - output_shape: Integer (max_sequence_length) for model output.
        Returns (None, None, None, None, None) if no data is processed.
    """
    specs_by_set = {"train": [], "val": [], "test": []}
    labels_by_set = {"train": [], "val": [], "test": []}
    max_len = 0
    num_features = None

    print("Processing data for datasets...")

    # --- Collect data from specified participants ---
    for set_name, participant_ids in [
        ("train", train_participant_ids),
        ("val", validation_participant_ids),
        ("test", test_participant_ids),
    ]:
        print(f"  Processing {set_name} set ({len(participant_ids)} participants)...")
        for p_id in participant_ids:
            if p_id not in processed_data:
                print(
                    f"    Warning: Participant {p_id} not found in processed_data. Skipping."
                )
                continue
            for side in ["left", "right"]:
                if side not in processed_data[p_id]:
                    # print(f"    Debug: Side {side} not found for participant {p_id}.") # Optional debug
                    continue
                if not processed_data[p_id][side]:
                    # print(f"    Debug: No recordings for {p_id} side {side}.") # Optional debug
                    continue

                for recording_idx, recording in enumerate(processed_data[p_id][side]):
                    # Spectrogram Sxx: shape (freq_bins, time_steps)
                    spectrogram = recording.get("spectrogram")
                    # Labels: shape (time_steps,)
                    labels = recording.get("labels")

                    if spectrogram is None or labels is None:
                        print(
                            f"    Warning: Missing 'spectrogram' or 'labels' for {p_id}/{side}/Rec{recording_idx}. Skipping."
                        )
                        continue

                    # Validate shapes: Spectrogram time dimension should match labels length
                    if spectrogram.shape[1] != len(labels):
                        print(
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
        print("Error: No valid data found to create datasets.")
        return None, None, None, None, None

    print(f"\nFound {num_features} frequency bins (features).")
    print(f"Maximum sequence length across all sets: {max_len}.")

    # --- Padding Sequences ---
    print("Padding sequences...")
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
                value=padding_value,
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
        print(
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

    print(
        f"  Padded Training data shape: X={X_train_padded.shape}, y={y_train_padded.shape}"
    )
    print(
        f"  Padded Validation data shape: X={X_val_padded.shape}, y={y_val_padded.shape}"
    )
    print(f"  Padded Test data shape: X={X_test_padded.shape}, y={y_test_padded.shape}")

    # --- Create TensorFlow Datasets ---
    print("Creating TensorFlow Datasets...")

    train_dataset = create_tf_dataset(X_train_padded, y_train_padded, batch_size)
    val_dataset = create_tf_dataset(X_val_padded, y_val_padded, batch_size)
    test_dataset = create_tf_dataset(X_test_padded, y_test_padded, batch_size)

    # --- Determine Shapes ---
    # Input shape for the model is (time_steps, features)
    input_shape = (max_len, num_features)
    # Output shape is the sequence length for labels
    output_shape = max_len

    print("\nDataset creation complete.")
    print(f"  Input Shape for Model: {input_shape}")
    print(f"  Output Shape (Sequence Length): {output_shape}")

    return train_dataset, val_dataset, test_dataset, input_shape, output_shape
