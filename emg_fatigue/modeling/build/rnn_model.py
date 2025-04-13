from typing import List, Tuple

import tensorflow as tf
from loguru import logger

RNN_UNITS = 128
DROPOUT_RATE = 0.3


def build_lstm_model(
    input_shape: Tuple[int, int],
    test_id: List[str],
    padding_value: float,
    rnn_units: int = RNN_UNITS,
    dropout_rate: float = DROPOUT_RATE,
) -> Tuple[tf.keras.Model, str]:
    """
    Create a recurrent neural network model for EMG fatigue regression.

    Uses unidirectional LSTM layers suitable for processing sequences.
    Handles variable-length sequences using padding and masking.
    Outputs one prediction (fatigue level) per time step.

    Args:
        input_shape: Expected shape of input data (max_sequence_length, num_features).
                     The model uses (None, num_features) internally for flexibility.
        test_id: List of participant IDs in the test set, used to create a unique model name.
                 Must contain exactly one ID.
        rnn_units: Number of units in the LSTM layers.
        dropout_rate: Dropout rate for regularization.
        padding_value: Value used for padding in the input sequences.

    Returns:
        A tuple containing:
        - model: The compiled TensorFlow Keras model
        - model_name: String identifier for the model based on test participant

    Raises:
        ValueError: If test_id list contains more than one ID.
    """
    # --- Construct model name ---
    # Test ID is included to distinguish between models trained on different test sets when saved as .keras files.

    if len(test_id) > 1:
        raise ValueError(
            f"Expected exactly one test ID, but got {len(test_id)}. Please provide a list with a single test ID."
        )

    if len(test_id) == 0:
        raise ValueError(
            "test_id list cannot be empty. Please provide a list with a single test ID."
        )

    # Extract the test participant ID from the list
    test_participant = test_id[0]

    # Generate model name based on test_id
    model_name = f"lstm_{test_participant}"

    # --- Build model ---
    logger.info(f"Building model for test participant: {test_participant}")

    # Input shape is (max_sequence_length, num_features)
    # The model needs (None, num_features) to handle variable time steps.
    _, num_features = input_shape

    # Input layer: Accepts sequences of variable length (None) with num_features features.
    input_layer = tf.keras.layers.Input(
        shape=(None, num_features), name="input_spectrogram"
    )

    # Masking layer: Ignores time steps where all features match the padding_value.
    # This prevents padded values from influencing subsequent layers.
    masked_input = tf.keras.layers.Masking(mask_value=padding_value, name="masking")(
        input_layer
    )

    # First LSTM layer: Processes the sequence, returning the output for each time step.
    lstm1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, name="lstm_1")(
        masked_input
    )
    # Dropout for regularization
    dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(lstm1)

    # Second LSTM layer: Further processes the sequence from the first LSTM layer.
    lstm2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, name="lstm_2")(
        dropout1
    )
    dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")(lstm2)

    # TimeDistributed Dense layer: Applies a dense layer independently to each time step's output.
    td_dense1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation="relu"), name="timedist_dense_1"
    )(dropout2)
    dropout3 = tf.keras.layers.Dropout(dropout_rate, name="dropout_3")(td_dense1)

    # TimeDistributed Output layer: Predicts one value (fatigue level) for each time step.
    # Output shape here is (batch_size, time_steps, 1)
    td_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1), name="timedist_output"
    )(dropout3)

    # Squeeze layer: Removes the last dimension (1) to match the label shape.
    # Output shape becomes (batch_size, time_steps)
    output_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1), name="squeeze_output"
    )(td_output)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # --- Compile model for regression ---
    # The loss function (MSE) automatically handles the mask passed from the Masking layer,
    # ignoring padded time steps during loss calculation.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],  # Mean Absolute Error is a common regression metric
    )

    return model, model_name
