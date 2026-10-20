"""Shared paths, log list, model tags and tolerance grid."""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
THP_DATA = os.path.join(DATA, "thp")                 # prefix-expanded THP splits (CSV)
S2S_DATA = os.path.join(DATA, "seq2seq")             # SuTraN-benchmark tensors per log
CKPT = os.path.join(ROOT, "checkpoints")
THP_CKPT = os.path.join(CKPT, "thp")
S2S_CKPT = os.path.join(CKPT, "seq2seq")
RESULTS = os.environ.get("JPA_RESULTS", os.path.join(ROOT, "results"))
KEYS = os.path.join(RESULTS, "keys")                 # (case, prefix length) per THP row
POP = os.path.join(RESULTS, "population")            # prefix-level intersection S
PRED = os.path.join(RESULTS, "predictions")          # per-instance predictions
TABLES = os.path.join(ROOT, "tables")
REFERENCE = os.path.join(ROOT, "reference", "paper_tables.json")

# Paper order (Table 1): ascending share of prefixes inside eps*.
LOGS = ["UQ_BPIC20RFP", "UQ_BPIC20DD", "UQ_HelpDesk", "UQ_BPIC20ID", "UQ_BPIC20TPD",
        "UQ_BPIC20PTC", "UQ_BPIC13I", "UQ_Sepsis", "UQ_BPIC12", "UQ_BPIC15_1"]
SHORT = {d: d[3:].replace("_", "-") for d in LOGS}

# Per-instance prediction files: results/predictions/<log>/<tag>.npz
TAGS = ["thp_mixture", "thp_baseline", "thp_baseline_zilm", "sutran", "edlstm"]
MODELS = ["THP-M", "THP-B", "SuTraN", "ED-LSTM"]

# Tolerance grid in days: 25 values, 86.4 ms .. 5 d; eps* = log-midpoint (193 s).
EPS_GRID = np.geomspace(1e-6, 5.0, 25)
EPS_STAR_IDX = 12
EPS_STAR = float(EPS_GRID[EPS_STAR_IDX])
SEC_PER_DAY = 86400.0
