
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from pathlib import Path
from typing import Dict, List, Tuple

from transformer.Constants import PAD 

# Raw CSV → Sequential Event Stream (SES) dictionary


CASE_COL  = "CaseID"
TIME_COL  = "Timestamp"
ACT_COL   = "Activity"
RT_COL    = "remaining_time"

SECONDS_PER_DAY = 86_400.0


def load_dataframes(
    directory: str,
    fold_filename: str,
    extension: str = ".csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three split CSVs from *directory*.

    Args:
        directory:      Folder containing the CSV files.
        fold_filename:  Stem used for train_/val_/test_ prefixes.
        extension:      File extension (default ``.csv``).

    Returns:
        ``(train_df, val_df, test_df)`` as DataFrames with
        their timestamp column already converted to ``datetime64``.
    """
    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(directory, name + extension)
        df = pd.read_csv(path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        return df

    train_df     = _read(f"train_{fold_filename}")
    val_df       = _read(f"val_{fold_filename}")
    test_df      = _read(f"test_{fold_filename}")
    return train_df, val_df, test_df


def build_event_type_map(train_df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
    """Build a deterministic activity-name → integer index mapping.

    The mapping is derived from the **training** set only to avoid
    information leakage from val/test partitions.

    Args:
        train_df:  Training-split DataFrame.

    Returns:
        ``(event_types, dim_process)`` where *event_types* maps each activity
        name to a zero-based integer and *dim_process* is the vocabulary size.
    """
    unique_acts = np.unique(train_df[ACT_COL])
    event_types = {name: idx for idx, name in enumerate(unique_acts)}
    return event_types, unique_acts.size


def add_time_features(df: pd.DataFrame, event_types: Dict[str, int]) -> pd.DataFrame:
    """Add engineered time and activity features to *df* in-place.

    New columns:
    - ``time_since_start``      – seconds from case start → days (float32)
    - ``time_since_last_event`` – inter-event duration → days (float32)
    - ``remaining_time``        – time until case end → days (float32)
    - ``type_event``            – integer activity code (unknown → -1)

    If ``CaseEndTime`` column is present (from prefix extraction), it is used
    as the case end reference for remaining_time instead of the last event in
    the prefix.

    Args:
        df:           Input event-log DataFrame (modified in-place).
        event_types:  Mapping from activity name to integer index.

    Returns:
        The modified DataFrame.
    """
    g = df.groupby(CASE_COL, sort=False)

    df["time_since_start"] = (
        (df[TIME_COL] - g[TIME_COL].transform("min"))
        .dt.total_seconds()
        .div(SECONDS_PER_DAY)
        .astype("float32")
    )
    df["time_since_last_event"] = (
        (df[TIME_COL] - g[TIME_COL].shift(1))
        .fillna(pd.Timedelta(0))
        .dt.total_seconds()
        .div(SECONDS_PER_DAY)
        .astype("float32")
    )

    if "CaseEndTime" in df.columns:
        case_end = pd.to_datetime(df["CaseEndTime"])
        df[RT_COL] = (
            (case_end - df[TIME_COL])
            .dt.total_seconds()
            .div(SECONDS_PER_DAY)
            .astype("float32")
        )
    else:
        df[RT_COL] = (
            (g[TIME_COL].transform("max") - df[TIME_COL])
            .dt.total_seconds()
            .div(SECONDS_PER_DAY)
            .astype("float32")
        )

    act_codes = df[ACT_COL].map(event_types)
    n_unknown = act_codes.isna().sum()
    if n_unknown > 0:
        print(f"  [Warning] {n_unknown} events with unseen activity mapped to UNK")
        # map unknown activities to a dedicated UNK index (max_known + 1)
        # so they do NOT collide with PAD=0 after the +1 shift in EventData
        unk_idx = len(event_types)
        act_codes = act_codes.fillna(unk_idx)
    df["type_event"] = act_codes.astype(int)

    return df


def df_to_ses(df: pd.DataFrame) -> List[List[Dict]]:
    """Convert a feature-enriched DataFrame into a list of event sequences.

    Each case becomes a list of event dictionaries with keys:
    ``time_since_start``, ``time_since_last_event``, ``remaining_time``,
    ``type_event``.

    Args:
        df:  DataFrame with columns added by :func:`add_time_features`.

    Returns:
        List of event sequences (Sequential Event Streams).
    """
    data = []
    for _, grp in df.groupby(CASE_COL, sort=False):
        seq = [
            {
                "time_since_start":      float(t0),
                "time_since_last_event": float(dt),
                "remaining_time":        float(rt),
                "type_event":            int(act),
            }
            for t0, dt, rt, act in zip(
                grp["time_since_start"].values,
                grp["time_since_last_event"].values,
                grp["remaining_time"].values,
                grp["type_event"].values,
            )
        ]
        data.append(seq)
    return data


def df_to_dict(
    directory: str,
    fold_filename: str,
    full_filename: str = None,
    extension: str = ".csv",
) -> Tuple[dict, dict, dict]:
    """End-to-end pipeline: CSVs → SES dictionaries ready for DataLoaders.

    Args:
        directory:      Folder containing the CSV splits.
        fold_filename:  Stem used for train_/val_/test_ prefixes.
        full_filename:  Unused, kept for backwards compatibility.
        extension:      File extension (default ``.csv``).

    Returns:
        ``(train_out, val_out, test_out)`` – each is a dict with keys
        ``dim_process``, ``max_length``, and the split data list under
        ``"train"``, ``"val"``, or ``"test"`` respectively.
    """
    train_df, val_df, test_df = load_dataframes(
        directory, fold_filename, extension
    )
    event_types, dim_process = build_event_type_map(train_df)

    max_len = 0
    has_unk = False
    for df in (train_df, val_df, test_df):
        add_time_features(df, event_types)
        if (df["type_event"] == len(event_types)).any():
            has_unk = True
        case_max = df.groupby(CASE_COL, sort=False).size().max()
        if case_max > max_len:
            max_len = case_max
    if has_unk:
        dim_process += 1  # account for UNK type in embedding

    train_ses = df_to_ses(train_df)
    val_ses   = df_to_ses(val_df)
    test_ses  = df_to_ses(test_df)

    # Per-case binary outcome labels (only present for the outcome-prediction
    # task). Order matches df_to_ses (groupby CASE_COL, sort=False); None when
    # the split has no ``label`` column (next-event / time tasks).
    train_lab = _extract_labels(train_df)
    val_lab   = _extract_labels(val_df)
    test_lab  = _extract_labels(test_df)

    common = {"dim_process": dim_process, "max_length": max_len}
    return (
        {**common, "train": train_ses, "labels": train_lab},
        {**common, "val":   val_ses,   "labels": val_lab},
        {**common, "test":  test_ses,  "labels": test_lab},
    )


def _extract_labels(df: pd.DataFrame):
    """One scalar 0/1 outcome label per case, aligned with :func:`df_to_ses`.

    Returns ``None`` when the DataFrame has no ``label`` column.
    """
    if "label" not in df.columns:
        return None
    return [float(grp["label"].iloc[0]) for _, grp in df.groupby(CASE_COL, sort=False)]



class EventData(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping a list of Sequential Event Streams.

    Args:
        data:  List of event sequences produced by :func:`df_to_ses`.
    """

    def __init__(self, data: List[List[Dict]], labels=None) -> None:
        self.time     = [[e["time_since_start"]      for e in inst] for inst in data]
        self.time_gap = [[e["time_since_last_event"]  for e in inst] for inst in data]
        self.r_time   = [[e["remaining_time"]          for e in inst] for inst in data]
        # Activity indices are 1-based inside the model (0 is reserved for PAD)
        self.activity = [[e["type_event"] + 1          for e in inst] for inst in data]
        # Optional per-case binary outcome label (outcome-prediction task only)
        self.labels   = labels
        self.length   = len(data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.labels is None:
            return (
                self.time[idx],
                self.time_gap[idx],
                self.r_time[idx],
                self.activity[idx],
            )
        return (
            self.time[idx],
            self.time_gap[idx],
            self.r_time[idx],
            self.activity[idx],
            self.labels[idx],
        )


def _pad_time(insts: List[List[float]]) -> torch.Tensor:
    """Right-pad a list of float sequences to the maximum length in *insts*."""
    max_len = max(len(s) for s in insts)
    batch   = np.array([s + [PAD] * (max_len - len(s)) for s in insts])
    return torch.tensor(batch, dtype=torch.float32)


def _pad_type(insts: List[List[int]]) -> torch.Tensor:
    """Right-pad a list of integer sequences to the maximum length in *insts*."""
    max_len = max(len(s) for s in insts)
    batch   = np.array([s + [PAD] * (max_len - len(s)) for s in insts])
    return torch.tensor(batch, dtype=torch.long)


def collate_fn(insts):
    """Custom collate function: pads variable-length sequences in a batch.

    Returns a 4-tuple for the next-event / time tasks, or a 5-tuple with a
    per-case ``labels`` float tensor ``[B]`` when the dataset carries outcome
    labels (auto-detected from the per-instance tuple length).
    """
    if len(insts[0]) == 5:
        time, time_gap, rt, activity, labels = zip(*insts)
        return (
            _pad_time(time),
            _pad_time(time_gap),
            _pad_time(rt),
            _pad_type(activity),
            torch.tensor(labels, dtype=torch.float32),
        )
    time, time_gap, rt, activity = zip(*insts)
    return (
        _pad_time(time),
        _pad_time(time_gap),
        _pad_time(rt),
        _pad_type(activity),
    )


def get_dataloader(
    data: List[List[Dict]],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
    labels=None,
) -> torch.utils.data.DataLoader:
    """Wrap an SES list in a :class:`torch.utils.data.DataLoader`.

    Args:
        data:         List of event sequences.
        batch_size:   Mini-batch size.
        shuffle:      Whether to shuffle before each epoch.
        num_workers:  Number of worker processes for data loading.

    Returns:
        A configured :class:`~torch.utils.data.DataLoader`.
    """
    dataset = EventData(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )