from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


DEFAULT_VELOCITY_CSV = "data/velocity_PID.csv"
DEFAULT_ACCEL_CSV = "data/accel_PID.csv"
DEFAULT_NREL_CSV = "data/FTP75_noisyconstantvel_PER1_movmedian_window15.csv"


def _read_second_column(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {csv_path}, found {df.shape[1]}")
    return df.iloc[:, 1].to_numpy()



def load_speed_profile(velocity_csv: Path) -> np.ndarray:
    velocity = _read_second_column(velocity_csv)
    velocity = velocity[1:]  # preserve original code behavior
    return np.asarray(velocity, dtype=np.float32)



def load_accel_profile(accel_csv: Path) -> np.ndarray:
    accel = _read_second_column(accel_csv)
    accel = accel[1:]  # preserve original code behavior
    return np.asarray(accel, dtype=np.float32)



def load_nrel_profile(nrel_csv: Path) -> pd.DataFrame:
    nrel = pd.read_csv(nrel_csv)
    nrel.iloc[:, 0] = nrel.iloc[:, 0] / 2.237
    return nrel.iloc[1:].reset_index(drop=True)



def trim_speed_profile(speed_profile: np.ndarray, max_episode_steps: int) -> np.ndarray:
    usable_steps = (len(speed_profile) // max_episode_steps) * max_episode_steps
    return speed_profile[:usable_steps]



def chunk_into_episodes(trimmed_profile: np.ndarray, max_episode_steps: int) -> List[np.ndarray]:
    return [
        trimmed_profile[start : start + max_episode_steps]
        for start in range(0, len(trimmed_profile), max_episode_steps)
    ]
