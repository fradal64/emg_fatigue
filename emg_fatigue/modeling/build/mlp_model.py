from typing import List, Tuple

import tensorflow as tf
from loguru import logger

HIDDEN_UNITS = 128
DROPOUT_RATE = 0.3


def build_mlp_model(
    input_shape: Tuple[int, int],
    test_id: List[str],
    padding_value: float,
    hidden_units: int = HIDDEN_UNITS,  # Number of units in hidden dense layers
    dropout_rate: float = DROPOUT_RATE,
) -> Tuple[tf.keras.Model, str]:
    """
    Create a linear feedforward neural network model for EMG fatigue regression.

    Uses TimeDistributed Dense layers to process sequences and make predictions
    at each time step, analogous to the LSTM model structure but without recurrence.
    Handles variable-length sequences using padding and masking.
    Outputs one prediction (fatigue level) per time step.

    Args:
        input_shape: Expected shape of input data (max_sequence_length, num_features).
                     The model uses (None, num_features) internally for flexibility.
        test_id: List of participant IDs in the test set, used to create a unique model name.
                 Must contain exactly one ID.
        padding_value: Value used for padding in the input sequences.
        hidden_units: Number of units in the hidden TimeDistributed Dense layers.
        dropout_rate: Dropout rate for regularization.

    Returns:
        A tuple containing:
        - model: The compiled TensorFlow Keras model
        - model_name: String identifier for the model based on test participant

    Raises:
        ValueError: If test_id list does not contain exactly one ID.
    """
    # --- Construct model name ---
    if len(test_id) != 1:
        raise ValueError(
            f"Expected exactly one test ID, but got {len(test_id)}. Please provide a list with a single test ID."
        )

    test_participant = test_id[0]
    model_name = f"mlp_{test_participant}"

    # --- Build model ---
    logger.info(f"Building MLP model for test participant: {test_participant}")

    _, num_features = input_shape

    input_layer = tf.keras.layers.Input(
        shape=(None, num_features), name="input_spectrogram"
    )

    masked_input = tf.keras.layers.Masking(mask_value=padding_value, name="masking")(
        input_layer
    )

    # First TimeDistributed Dense layer
    td_dense1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_units, activation="relu"), name="timedist_dense_1"
    )(masked_input)
    dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(td_dense1)

    # Second TimeDistributed Dense layer
    td_dense2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(hidden_units, activation="relu"), name="timedist_dense_2"
    )(dropout1)
    dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(td_dense2)

    # TimeDistributed Output layer: Predicts one value (fatigue level) between 0 and 1 for each time step.
    td_sigmoid_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation="sigmoid"), name="timedist_sigmoid_output"
    )(dropout2)

    # Scale output to be between 0 and 100
    scaled_output = tf.keras.layers.Lambda(lambda x: x * 100.0, name="scale_output")(
        td_sigmoid_output
    )

    # Squeeze layer: Removes the last dimension (1) to match the label shape.
    output_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="squeeze_output"
    )(scaled_output)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # --- Compile model for regression ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model, model_name
