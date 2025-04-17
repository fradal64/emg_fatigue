import math
from typing import List, Tuple

import tensorflow as tf
from loguru import logger

# Configuration constants (can be adjusted)
NUM_HEADS = 8  # Number of attention heads
FF_DIM = 512  # Hidden layer size in feed forward network inside transformer
NUM_TRANSFORMER_BLOCKS = 4  # Number of transformer blocks
D_MODEL = 256  # Embedding dimension (must be divisible by NUM_HEADS)
DROPOUT_RATE = 0.1  # Dropout rate


# Positional Encoding
# Transformers need positional information since they don't process sequences step-by-step.
def get_positional_encoding(seq_len, d_model):
    """Generates positional encoding sinusoids."""
    # Ensure d_model is even for sin/cos pairing
    if d_model % 2 != 0:
        raise ValueError(
            f"Cannot create sinusoidal positional encoding with odd d_model ({d_model})."
            " d_model must be even."
        )

    positions = tf.range(seq_len, dtype=tf.float32)[
        :, tf.newaxis
    ]  # Shape: (seq_len, 1)

    # Calculate the division term for the frequencies
    # Correct calculation for indices 0, 2, 4, ... up to d_model
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model)
    )  # Shape: (d_model / 2,)

    # Calculate sin values for even indices (0, 2, ...)
    sin_vals = tf.sin(positions * div_term)  # Shape: (seq_len, d_model / 2)
    # Calculate cos values for odd indices (1, 3, ...) - uses the same div_term
    cos_vals = tf.cos(positions * div_term)  # Shape: (seq_len, d_model / 2)

    # Interleave sin and cos values
    # Stack along a new last dimension -> shape (seq_len, d_model / 2, 2)
    pos_encoding = tf.stack([sin_vals, cos_vals], axis=-1)

    # Reshape to (seq_len, d_model)
    pos_encoding = tf.reshape(pos_encoding, [seq_len, d_model])

    pos_encoding = pos_encoding[
        tf.newaxis, ...
    ]  # Add batch dimension: (1, seq_len, d_model)
    return tf.cast(pos_encoding, dtype=tf.float32)


# Transformer Encoder Block
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Represents one block of the Transformer encoder."""

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        use_causal_mask=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_causal_mask = use_causal_mask
        self.supports_masking = True

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            name="multi_head_attention",
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
            name="feed_forward",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_1"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="layer_norm_2"
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")

    def call(self, inputs, training=False, mask=None):
        # Multi-head self-attention
        # Keras automatically passes the computed mask to the attention layer
        # as attention_mask if the layer supports it and inputs have a mask.
        # Let Keras handle mask combination internally when use_causal_mask is True
        # attention_mask = mask # Explicitly use the input mask for clarity if needed by MHA
        if self.use_causal_mask:
            # When use_causal_mask=True, MHA automatically combines the input mask
            # (from inputs._keras_mask) with the generated causal mask.
            # Do not pass attention_mask explicitly here.
            attn_output = self.att(
                query=inputs, value=inputs, key=inputs, use_causal_mask=True
            )
        else:
            # If not using causal mask, pass the padding mask explicitly
            attn_output = self.att(
                query=inputs,
                value=inputs,
                key=inputs,
                attention_mask=mask,  # Pass the padding mask
                use_causal_mask=False,
            )

        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection + Layer Normalization
        out1 = self.layernorm1(inputs + attn_output)

        # Feed Forward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection + Layer Normalization
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def compute_mask(self, inputs, mask=None):
        # Pass the input mask through unchanged
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
                "use_causal_mask": self.use_causal_mask,
            }
        )
        return config


# Positional Encoding Layer
class PositionalEncodingLayer(tf.keras.layers.Layer):
    """Adds positional encoding to the input embeddings."""

    def __init__(self, max_seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
        self.supports_masking = True

    def call(self, x, mask=None):
        # Input x shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        # Slice positional encoding to match input seq_len and add to input
        return x + self.pos_encoding[:, :seq_len, :]

    def compute_mask(self, inputs, mask=None):
        # Pass the input mask through unchanged
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_seq_len": self.max_seq_len,
                "d_model": self.d_model,
            }
        )
        return config


def build_transformer_model(
    input_shape: Tuple[int, int],
    test_id: List[str],
    padding_value: float,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    ff_dim: int = FF_DIM,
    num_transformer_blocks: int = NUM_TRANSFORMER_BLOCKS,
    dropout_rate: float = DROPOUT_RATE,
) -> Tuple[tf.keras.Model, str]:
    """
    Creates a Transformer encoder model for EMG fatigue regression.

    Uses Transformer encoder blocks suitable for processing sequences.
    Handles variable-length sequences using padding and masking.
    Includes positional encoding.
    Outputs one prediction (fatigue level) per time step.

    Args:
        input_shape: Expected shape of input data (max_sequence_length, num_features).
        test_id: List of participant IDs in the test set, used to create a unique model name.
                 Must contain exactly one ID.
        padding_value: Value used for padding in the input sequences.
        d_model: Embedding dimension for the Transformer (must be divisible by num_heads).
        num_heads: Number of attention heads in MultiHeadAttention layers.
        ff_dim: Hidden layer size in the feed-forward networks within Transformer blocks.
        num_transformer_blocks: Number of Transformer encoder blocks to stack.
        dropout_rate: Dropout rate for regularization.

    Returns:
        A tuple containing:
        - model: The compiled TensorFlow Keras model
        - model_name: String identifier for the model based on test participant

    Raises:
        ValueError: If test_id list does not contain exactly one ID.
        ValueError: If d_model is not divisible by num_heads.
    """
    # --- Construct model name ---
    if len(test_id) != 1:
        raise ValueError(
            f"Expected exactly one test ID, but got {len(test_id)}. Please provide a list with a single test ID."
        )
    test_participant = test_id[0]
    model_name = f"transformer_{test_participant}"

    # --- Validate configuration ---
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
        )

    # --- Build model ---
    logger.info(f"Building Transformer model for test participant: {test_participant}")

    max_sequence_length, num_features = input_shape

    # Input layer: Accepts sequences of variable length (None) with num_features features.
    input_layer = tf.keras.layers.Input(
        shape=(None, num_features), name="input_spectrogram"
    )

    # Masking layer: Ignores time steps where all features match the padding_value.
    masked_input = tf.keras.layers.Masking(mask_value=padding_value, name="masking")(
        input_layer
    )

    # Dense layer to project input features to d_model dimensions
    # This is often needed if num_features != d_model
    if num_features != d_model:
        projected_input = tf.keras.layers.Dense(d_model, name="input_projection")(
            masked_input
        )
    else:
        projected_input = masked_input  # No projection needed if dimensions match

    # Add positional encoding
    # Use max_sequence_length from input_shape for the encoding generation
    # The layer will handle slicing if actual sequence length is shorter
    pos_encoded_input = PositionalEncodingLayer(
        max_sequence_length, d_model, name="positional_encoding"
    )(projected_input)

    # Dropout after positional encoding
    x = tf.keras.layers.Dropout(dropout_rate, name="pos_encoding_dropout")(
        pos_encoded_input
    )

    # Stacked Transformer Encoder Blocks
    for i in range(num_transformer_blocks):
        x = TransformerEncoderBlock(
            d_model,
            num_heads,
            ff_dim,
            dropout_rate,
            name=f"transformer_block_{i + 1}",
            use_causal_mask=True,
        )(x)  # Pass mask implicitly

    # TimeDistributed Dense layer: Applies a dense layer independently to each time step's output.
    td_dense1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(32, activation="relu"), name="timedist_dense_1"
    )(x)
    dropout_final = tf.keras.layers.Dropout(dropout_rate, name="final_dropout")(
        td_dense1
    )

    # TimeDistributed Output layer: Predicts one value (fatigue level) for each time step.
    td_output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1), name="timedist_output"
    )(dropout_final)

    # Squeeze layer: Removes the last dimension (1) to match the label shape.
    output_layer = tf.keras.layers.Lambda(
        lambda t: tf.squeeze(t, axis=-1), name="squeeze_output"
    )(td_output)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # --- Compile model for regression ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model, model_name


# Register custom layers for saving/loading
tf.keras.utils.get_custom_objects().update(
    {
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "PositionalEncodingLayer": PositionalEncodingLayer,
    }
)
