from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from tabulate import tabulate


def analyze_label_slopes(
    processed_data: Dict[str, Dict[str, List[Dict[str, np.ndarray]]]],
) -> Optional[Dict[str, Any]]:
    """
    Analyzes the slopes of the true fatigue labels across all recordings and
    calculates baseline MAE using mean/median slopes with leave-one-out cross-validation.

    Args:
        processed_data: Dictionary containing processed data for all participants.
                       Structure: {p_id: {'left': [rec_dict, ...], 'right': [...]}}
                       Each rec_dict must contain 'time_vector' and 'labels'.

    Returns:
        A dictionary containing slope statistics and baseline MAE values, or None
        if no valid slopes could be calculated. Structure:
        {
            "slope_stats": {
                'mean': float, 'median': float, 'std': float,
                'min': float, 'max': float, 'count': int
            },
            "slopes_by_participant": Dict[str, Dict[str, List[float]]],
            "loocv_results": Dict[str, Dict[str, float]] # MAE per participant
        }
    """
    slopes_by_participant: Dict[str, Dict[str, List[float]]] = {}
    all_slopes: List[float] = []

    # For storing Leave-One-Out Cross Validation results
    loocv_results: Dict[str, Dict[str, float]] = {}

    logger.info("Analyzing true fatigue label slopes across all recordings...")

    # --- Pass 1: Calculate slopes for each recording ---
    for p_id, sides_data in processed_data.items():
        slopes_by_participant.setdefault(p_id, {"left": [], "right": []})
        for side, recordings in sides_data.items():
            if side not in ["left", "right"]:
                continue  # Skip if side key isn't left or right

            for rec_idx, recording in enumerate(recordings):
                t_vector = recording.get("time_vector")
                y_true = recording.get("labels")

                # Validate data needed for slope calculation
                if (
                    t_vector is None
                    or y_true is None
                    or len(t_vector) < 2
                    or len(t_vector) != len(y_true)
                ):
                    continue

                try:
                    # Perform linear regression: y = slope * x + intercept
                    slope, intercept, _, _, _ = stats.linregress(t_vector, y_true)
                    if np.isnan(slope):
                        continue
                    slopes_by_participant[p_id][side].append(slope)
                    all_slopes.append(slope)
                except ValueError as e:
                    logger.warning(
                        f"Linregress failed for {p_id}/{side}/Rec{rec_idx + 1}: {e}"
                    )

    if not all_slopes:
        logger.error(
            "No valid slopes calculated. Cannot provide statistics or baseline MAE."
        )
        return None

    # --- Calculate Overall Slope Statistics ---
    slope_stats = {
        "mean": np.mean(all_slopes),
        "median": np.median(all_slopes),
        "std": np.std(all_slopes),
        "min": np.min(all_slopes),
        "max": np.max(all_slopes),
        "count": len(all_slopes),
    }
    logger.info(
        f"Overall True Label Slope Stats (n={slope_stats['count']}): "
        f"Mean={slope_stats['mean']:.4f}, Median={slope_stats['median']:.4f}, "
        f"Std={slope_stats['std']:.4f}"
    )

    # --- Pass 2: Leave-One-Out Cross Validation MAE ---
    logger.info("Performing leave-one-out evaluation of baseline linear models...")

    # Create flattened list of participant-slope pairs for easier filtering
    participant_slopes: List[Tuple[str, float]] = []
    for p_id, sides in slopes_by_participant.items():
        for side_slopes in sides.values():
            for slope in side_slopes:
                participant_slopes.append((p_id, slope))

    # For each participant, perform leave-one-out prediction
    for test_p_id, sides_data in processed_data.items():
        # Skip if no slopes for this participant
        if test_p_id not in slopes_by_participant or not any(
            slopes_by_participant[test_p_id].values()
        ):
            continue

        # Calculate leave-one-out slopes (excluding the test participant)
        loo_slopes = [
            slope for (p_id, slope) in participant_slopes if p_id != test_p_id
        ]

        if not loo_slopes:
            logger.warning(
                f"No other participant slopes available to evaluate {test_p_id}. Skipping."
            )
            continue

        loo_mean_slope = np.mean(loo_slopes)
        loo_median_slope = np.median(loo_slopes)

        # Initialize error metrics for this participant
        errors_mean: List[float] = []
        errors_median: List[float] = []

        # Calculate prediction errors using leave-one-out slopes
        for side, recordings in sides_data.items():
            if side not in ["left", "right"]:
                continue

            for rec_idx, recording in enumerate(recordings):
                t_vector = recording.get("time_vector")
                y_true = recording.get("labels")

                # Skip if data is missing or empty
                if t_vector is None or y_true is None or len(t_vector) == 0:
                    continue

                # Predict using leave-one-out mean slope
                y_pred_mean = loo_mean_slope * t_vector
                abs_errors_mean = np.abs(y_true - y_pred_mean)
                errors_mean.extend(abs_errors_mean)

                # Predict using leave-one-out median slope
                y_pred_median = loo_median_slope * t_vector
                abs_errors_median = np.abs(y_true - y_pred_median)
                errors_median.extend(abs_errors_median)

        # Calculate MAE for this participant
        if errors_mean and errors_median:
            mae_mean = np.mean(errors_mean)
            mae_median = np.mean(errors_median)

            # Store results
            loocv_results[test_p_id] = {
                "mae_mean_slope": mae_mean,
                "mae_median_slope": mae_median,
                "mean_slope_used": loo_mean_slope,
                "median_slope_used": loo_median_slope,
                "num_data_points": len(errors_mean),
            }

            logger.info(
                f"Participant {test_p_id} - MAE using leave-one-out mean slope: {mae_mean:.4f}, "
                f"MAE using leave-one-out median slope: {mae_median:.4f}"
            )

    # --- Return Results ---
    results = {
        "slope_stats": slope_stats,
        "slopes_by_participant": slopes_by_participant,
        "loocv_results": loocv_results,
    }

    # Calculate and log average LOOCV MAE
    if loocv_results:
        avg_mae_mean = np.mean(
            [res["mae_mean_slope"] for res in loocv_results.values()]
        )
        avg_mae_median = np.mean(
            [res["mae_median_slope"] for res in loocv_results.values()]
        )

        logger.info(f"Average LOOCV MAE using mean slope: {avg_mae_mean:.4f}")
        logger.info(f"Average LOOCV MAE using median slope: {avg_mae_median:.4f}")

        results["avg_loocv_mae_mean_slope"] = avg_mae_mean
        results["avg_loocv_mae_median_slope"] = avg_mae_median

    return results


def print_loocv_results_table(results: Dict[str, Any]) -> None:
    """
    Prints a formatted table of the LOOCV results for each participant.

    Args:
        results: The dictionary returned by analyze_label_slopes
    """
    if "loocv_results" not in results or not results["loocv_results"]:
        logger.warning("No LOOCV results available to print.")
        return

    # Prepare data for pandas DataFrame
    table_data = []
    for p_id, metrics in results["loocv_results"].items():
        table_data.append(
            {
                "Participant": p_id,
                "MAE (Mean Slope)": metrics["mae_mean_slope"],
                "MAE (Median Slope)": metrics["mae_median_slope"],
                "Mean Slope Used": metrics["mean_slope_used"],
                "Median Slope Used": metrics["median_slope_used"],
                "Data Points": metrics["num_data_points"],
            }
        )

    # Create DataFrame and add average row
    df = pd.DataFrame(table_data)
    avg_row = pd.DataFrame(
        [
            {
                "Participant": "AVERAGE",
                "MAE (Mean Slope)": results.get("avg_loocv_mae_mean_slope", np.nan),
                "MAE (Median Slope)": results.get("avg_loocv_mae_median_slope", np.nan),
                "Mean Slope Used": np.nan,
                "Median Slope Used": np.nan,
                "Data Points": df["Data Points"].sum() if not df.empty else 0,
            }
        ]
    )

    df = pd.concat([df, avg_row], ignore_index=True)

    # Sort by participant ID, keeping AVERAGE at the bottom
    if not df.empty:
        df = pd.concat(
            [
                df[df["Participant"] != "AVERAGE"].sort_values("Participant"),
                df[df["Participant"] == "AVERAGE"],
            ]
        ).reset_index(drop=True)

    # Print table
    print("\n--- Leave-One-Out Cross Validation Results ---")
    print(
        tabulate(df, headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f")
    )
    print(f"\nOverall Statistics (n={results['slope_stats']['count']}):")
    print(
        f"All slopes - Mean: {results['slope_stats']['mean']:.4f}, Median: {results['slope_stats']['median']:.4f}, Std: {results['slope_stats']['std']:.4f}"
    )
    print(
        f"Range: [{results['slope_stats']['min']:.4f}, {results['slope_stats']['max']:.4f}]"
    )


# Example usage for notebook
"""
from emg_fatigue.analysis.slope_analysis import analyze_label_slopes, print_loocv_results_table

# Assuming processed_data is already loaded
analysis_results = analyze_label_slopes(processed_data)
if analysis_results:
    print_loocv_results_table(analysis_results)
"""
