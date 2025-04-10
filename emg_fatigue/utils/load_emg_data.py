import glob
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from emg_fatigue.config import TRIMMED_DATA_DIR


def load_all_participant_data(
    data_dir: Path = TRIMMED_DATA_DIR,
) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
    """
    Load all participant data into a structured dictionary.

    Args:
        data_dir: Path to the directory containing participant data

    Returns:
        Dictionary with structure:
        {
            'P001': {
                'left': [df1, df2, ...],  # DataFrames for left bicep
                'right': [df1, df2, ...], # DataFrames for right bicep
            },
            'P002': {...}
        }
    """
    participant_data = {}
    participant_dirs = sorted(glob.glob(str(data_dir / "P*")))

    for p_dir in participant_dirs:
        participant_id = os.path.basename(p_dir)
        participant_data[participant_id] = {"left": [], "right": []}

        # Get left and right bicep folders - assuming one folder of each type exists
        left_folder = next(Path(p_dir).glob("*Left"))
        right_folder = next(Path(p_dir).glob("*Right"))

        # Process left bicep data
        left_files = sorted(list(left_folder.glob("*.csv")))
        for file in left_files:
            df = load_emg_file(file)
            participant_data[participant_id]["left"].append(df)

        # Process right bicep data
        right_files = sorted(list(right_folder.glob("*.csv")))
        for file in right_files:
            df = load_emg_file(file)
            participant_data[participant_id]["right"].append(df)

    return participant_data


def load_emg_file(file_path):
    """Load a single sEMG file, extract relevant columns, and normalize time to start at 0."""
    # Read the file but handle potential trailing commas and extra columns
    df = pd.read_csv(
        file_path,
        usecols=["time_sEMG_seconds", "sEMG"],  # Only load the columns we need
        na_values=[""],  # Treat empty strings as NaN
        skip_blank_lines=True,
    )  # Skip blank lines

    # Drop rows where either time or EMG is NaN
    df_clean = df.dropna()

    # Normalize time to start at 0, if data exists
    if not df_clean.empty:
        # Use .loc to explicitly modify the DataFrame and avoid SettingWithCopyWarning
        df_clean.loc[:, "time_sEMG_seconds"] = (
            df_clean["time_sEMG_seconds"] - df_clean["time_sEMG_seconds"].iloc[0]
        )

    return df_clean  # Return only non-NaN rows
