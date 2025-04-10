from typing import Optional

import tensorflow as tf
from loguru import logger

from emg_fatigue.config import MODELS_DIR


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str,
    epochs: int = 100,
) -> Optional[tf.keras.callbacks.History]:
    """
    Trains a given Keras RNN model using provided datasets and callbacks.

    Args:
        model: The compiled Keras model to train.
        train_ds: The TensorFlow Dataset for training.
        val_ds: The TensorFlow Dataset for validation.
        model_name: Name of the model file without extension. Will be saved in MODELS_DIR.
        epochs: Maximum number of epochs to train for.

    Returns:
        The training history object, or None if training could not start.
    """
    # Fixed hyperparameters
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-6

    if train_ds is None or val_ds is None:
        logger.error("Training or validation dataset is None. Skipping training.")
        return None

    # Construct model save path using MODELS_DIR from config
    if not model_name.endswith((".h5", ".keras")):
        model_name += ".keras"  # Add extension if not provided

    model_save_path = MODELS_DIR / model_name

    # Ensure the directory for saving the model exists
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model will be saved to: {model_save_path}")
    except Exception as e:
        logger.error(f"Failed to create directory for model saving: {e}")
        return None

    # --- Define Callbacks ---
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --- Train the Model ---
    logger.info(f"Starting model training for up to {epochs} epochs...")
    try:
        # Train the model
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Training finished.")
        return history
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
        return None
