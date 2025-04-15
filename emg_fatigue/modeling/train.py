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


def fine_tune_model(
    model: tf.keras.Model,
    fine_tune_train_ds: tf.data.Dataset,
    fine_tune_val_ds: Optional[tf.data.Dataset],
    model_name: str,  # Base model name (e.g., "lstm_P001")
    fine_tune_epochs: int = 20,
    fine_tune_lr: float = 1e-4,
) -> Optional[tf.keras.callbacks.History]:
    """
    Fine-tunes a pre-trained Keras model on a specific fine-tuning dataset,
    using a dedicated validation set for the fine-tuning phase.

    Assumes the initial model weights are already loaded (e.g., via restore_best_weights=True
    in initial training or by loading the saved model).

    Args:
        model: The compiled Keras model with pre-trained weights.
        fine_tune_train_ds: The TensorFlow Dataset for fine-tuning training.
        fine_tune_val_ds: The TensorFlow Dataset for validation during fine-tuning (optional).
        model_name: Base name of the model (e.g., "lstm_P001"). The fine-tuned model
                    will be saved with a "_finetuned" suffix.
        fine_tune_epochs: Maximum number of epochs for fine-tuning.
        fine_tune_lr: Learning rate for the fine-tuning phase.

    Returns:
        The fine-tuning history object, or None if fine-tuning could not start.
    """
    # Fine-tuning specific hyperparameters
    FINE_TUNE_EARLY_STOPPING_PATIENCE = 5
    FINE_TUNE_REDUCE_LR_PATIENCE = 3
    FINE_TUNE_REDUCE_LR_FACTOR = 0.5
    FINE_TUNE_MIN_LR = 1e-7

    if fine_tune_train_ds is None:
        logger.error("Fine-tuning training dataset is None. Skipping fine-tuning.")
        return None

    # Construct fine-tuned model save path
    if model_name.endswith((".h5", ".keras")):
        base_name = model_name.rsplit(".", 1)[0]
    else:
        base_name = model_name
    fine_tuned_model_name = f"{base_name}_finetuned.keras"
    fine_tuned_model_save_path = MODELS_DIR / fine_tuned_model_name

    # Ensure the directory for saving the model exists
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Fine-tuned model will be saved to: {fine_tuned_model_save_path}")
    except Exception as e:
        logger.error(f"Failed to create directory for fine-tuned model saving: {e}")
        return None

    # --- Re-compile model with fine-tuning learning rate ---
    # It's crucial to re-compile to set the new learning rate
    logger.info(f"Re-compiling model with fine-tuning learning rate: {fine_tune_lr}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="mse",
        metrics=["mae"],
    )

    # --- Define Fine-tuning Callbacks ---
    # Use fine-tune validation set for monitoring if provided, otherwise monitor training loss
    if fine_tune_val_ds:
        monitor_metric = "val_loss"
        logger.info(
            f"Monitoring '{monitor_metric}' using fine_tune_val_ds for fine-tuning callbacks."
        )
    else:
        monitor_metric = "loss"
        logger.warning(
            f"No fine_tune_val_ds provided. Monitoring training '{monitor_metric}' for fine-tuning callbacks."
        )

    ft_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=FINE_TUNE_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=FINE_TUNE_REDUCE_LR_FACTOR,
            patience=FINE_TUNE_REDUCE_LR_PATIENCE,
            min_lr=FINE_TUNE_MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fine_tuned_model_save_path),
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1,
        ),
    ]

    # --- Fine-tune the Model ---
    logger.info(f"Starting model fine-tuning for up to {fine_tune_epochs} epochs...")
    try:
        history = model.fit(
            fine_tune_train_ds,
            epochs=fine_tune_epochs,
            validation_data=fine_tune_val_ds,
            callbacks=ft_callbacks,
            verbose=1,
        )
        logger.info("Fine-tuning finished.")
        return history
    except Exception as e:
        logger.error(f"An error occurred during model fine-tuning: {e}")
        return None
