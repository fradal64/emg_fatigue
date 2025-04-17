from datetime import datetime

import pandas as pd
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from emg_fatigue.config import BATCH_SIZE, PADDING_VALUE, RANDOM_SEED, REPORTS_DIR
from emg_fatigue.modeling.build.gru_model import build_gru_model
from emg_fatigue.modeling.build.mlp_model import build_mlp_model
from emg_fatigue.modeling.build.rnn_model import build_lstm_model
from emg_fatigue.modeling.build.transformer_model import build_transformer_model
from emg_fatigue.modeling.evaluate import evaluate_model
from emg_fatigue.modeling.train import fine_tune_model, train_model
from emg_fatigue.plots.visualize_model_predictions import visualize_model_predictions
from emg_fatigue.utils.create_loocv_dataset import create_loocv_dataset
from emg_fatigue.utils.load_emg_data import load_all_participant_data
from emg_fatigue.utils.process_emg_data import process_all_participant_data

# Define parameters
NUM_FINE_TUNING_RECS = 4
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 100
FINE_TUNE_LR = 1e-5

# Configure logger to write to a file
log_file_path = REPORTS_DIR / "loocv_run.log"
logger.add(log_file_path, rotation="1 MB", retention="10 days", level="INFO")


def run_model_training(
    processed_data,
    train_ids,
    val_ids,
    test_ids,
    model_builder,
    results_df,
    run_timestamp,
):
    """Run training for a specific model and participant split"""

    # Create dataset
    (
        train_ds,
        val_ds,
        fine_tune_train_ds,
        fine_tune_val_ds,
        test_ds,
        input_shape,
        output_shape,
        norm_mean,
        norm_std,
        test_recording_indices,
    ) = create_loocv_dataset(
        processed_data=processed_data,
        train_participant_ids=train_ids,
        validation_participant_ids=val_ids,
        test_participant_ids=test_ids,
        batch_size=BATCH_SIZE,
        padding_value=PADDING_VALUE,
        normalize=True,
        augment=True,
        num_fine_tuning_recordings_per_subject=NUM_FINE_TUNING_RECS,
    )

    # Check if datasets were created successfully
    if train_ds is None or val_ds is None or test_ds is None:
        logger.error(f"Failed to create datasets for test participant {test_ids}")
        return None

    # Build model
    model, model_name = model_builder(
        input_shape=input_shape, test_id=test_ids, padding_value=PADDING_VALUE
    )
    model_type = model_name.split("_")[0]  # Extract model type (lstm, mlp, etc.)

    logger.info(f"--- Starting Initial Training for {model_name} ---")
    initial_history = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=model_name,
        epochs=INITIAL_EPOCHS,
    )

    if initial_history is None:
        logger.error(f"Initial training failed for {model_name}")
        return None

    # Fine-tuning (Optional)
    fine_tuned_model_name_for_eval = model_name
    if fine_tune_train_ds is not None:
        logger.info(f"--- Starting Fine-tuning for {model_name} ---")

        fine_tune_history = fine_tune_model(
            model=model,
            fine_tune_train_ds=fine_tune_train_ds,
            fine_tune_val_ds=fine_tune_val_ds,
            model_name=model_name,
            fine_tune_epochs=FINE_TUNE_EPOCHS,
            fine_tune_lr=FINE_TUNE_LR,
        )

        if fine_tune_history is None:
            logger.warning(
                f"Fine-tuning failed or was skipped for {model_name}. Proceeding with initial model."
            )
        else:
            logger.info("Fine-tuning complete. Using fine-tuned model.")
            if model_name.endswith((".h5", ".keras")):
                base_name = model_name.rsplit(".", 1)[0]
            else:
                base_name = model_name
            fine_tuned_model_name_for_eval = f"{base_name}_finetuned"
    else:
        logger.info(
            f"--- Skipping Fine-tuning (no fine_tune_train_ds available) for {model_name} ---"
        )

    # Evaluation
    logger.info(f"--- Evaluating final model: {fine_tuned_model_name_for_eval} ---")
    final_metrics = evaluate_model(
        model=model,
        test_ds=test_ds,
        model_name=fine_tuned_model_name_for_eval,
        table_format="grid",
    )
    logger.info(f"Final Evaluation Metrics: {final_metrics}")

    # Visualize predictions
    logger.info(
        f"--- Visualizing predictions for model: {fine_tuned_model_name_for_eval} ---"
    )
    visualize_model_predictions(
        model=model,
        model_name=fine_tuned_model_name_for_eval,
        processed_data=processed_data,
        test_participant_ids=test_ids,
        input_shape=input_shape,
        norm_mean=norm_mean,
        norm_std=norm_std,
        test_recording_indices=test_recording_indices,
    )

    # Record results
    if final_metrics and len(test_ids) == 1:
        test_id = test_ids[0]
        result_row = {
            "test_participant": test_id,
            "model_type": model_type,
            "mae": final_metrics.get("mae", float("nan")),
            "mse": final_metrics.get("mse", float("nan")),
            "timestamp": run_timestamp,
        }

        results_df = pd.concat(
            [results_df, pd.DataFrame([result_row])], ignore_index=True
        )

    return results_df


def main():
    # Create a timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory using config paths
    loocv_results_dir = REPORTS_DIR / "loocv_results"
    loocv_results_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"LOOCV results will be saved to: {loocv_results_dir}")

    # Initialize results dataframe
    results_df = pd.DataFrame(
        columns=[
            "test_participant",
            "model_type",
            "mae",
            "mse",
            "timestamp",
        ]
    )

    # Load and process data
    logger.info("Loading data...")
    raw_data = load_all_participant_data()
    processed_data = process_all_participant_data(participant_data=raw_data)

    # Get all participant IDs
    all_participant_ids = list(processed_data.keys())
    logger.info(f"Found {len(all_participant_ids)} participants: {all_participant_ids}")

    # Model builders
    model_builders = [
        build_lstm_model,
        build_mlp_model,
        build_transformer_model,
        build_gru_model,
    ]

    # Iterate through all participants for LOOCV
    for test_id in tqdm(all_participant_ids, desc="LOOCV Progress"):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Starting LOOCV for test participant: {test_id}")
        logger.info(f"{'=' * 50}\n")

        # Rotate participants for train, validation, and test
        test_ids = [test_id]
        remaining_ids = [pid for pid in all_participant_ids if pid != test_id]

        # Rotate validation participants
        val_ids = remaining_ids[:2]  # Use the first two remaining as validation
        train_ids = remaining_ids[2:]  # Use the rest for training

        logger.info(f"Train participants: {train_ids}")
        logger.info(f"Validation participants: {val_ids}")
        logger.info(f"Test participant: {test_ids}")

        # Train all model types
        for model_builder in model_builders:
            try:
                model_type = model_builder.__name__.split("_")[
                    -2
                ]  # Extract model type from function name
                logger.info(f"\n{'-' * 50}")
                logger.info(
                    f"Training {model_type.upper()} model for test participant {test_id}"
                )
                logger.info(f"{'-' * 50}")

                results_df = run_model_training(
                    processed_data=processed_data,
                    train_ids=train_ids,
                    val_ids=val_ids,
                    test_ids=test_ids,
                    model_builder=model_builder,
                    results_df=results_df,
                    run_timestamp=run_timestamp,
                )

                # Save intermediate results after each model
                results_csv_path = (
                    loocv_results_dir / f"loocv_results_{run_timestamp}.csv"
                )
                results_df.to_csv(results_csv_path, index=False)
                logger.info(f"Updated results saved to {results_csv_path}")

            except Exception as e:
                logger.exception(
                    f"Error training {model_builder.__name__} for test participant {test_id}: {str(e)}"
                )

    # Generate summary statistics
    summary_df = results_df.groupby("model_type").agg(
        {
            "mae": ["mean", "std", "min", "max"],
            "mse": ["mean", "std", "min", "max"],
        }
    )

    # Save final results and summary
    final_results_path = loocv_results_dir / f"loocv_final_results_{run_timestamp}.csv"
    summary_path = loocv_results_dir / f"loocv_summary_{run_timestamp}.csv"

    results_df.to_csv(final_results_path, index=False)
    summary_df.to_csv(summary_path)

    # Also save summary as a pretty table for the paper
    with open(loocv_results_dir / f"loocv_summary_table_{run_timestamp}.txt", "w") as f:
        f.write(
            f"Model Performance Summary (LOOCV, {len(all_participant_ids)} participants):\n\n"
        )
        f.write(summary_df.to_string())

    logger.info(f"\n{'=' * 50}")
    logger.info("LOOCV completed successfully!")
    logger.info(f"Final results saved to {final_results_path}")
    logger.info(f"Summary statistics saved to {summary_path}")
    logger.info(f"{'=' * 50}")

    # Print summary table
    logger.info("\nModel Performance Summary:")
    logger.info(summary_df.to_string())


if __name__ == "__main__":
    try:
        # Set TF seed for reproducibility
        tf.random.set_seed(RANDOM_SEED)
        main()
    except Exception as e:
        logger.exception(f"Error in main execution: {str(e)}")
        raise
