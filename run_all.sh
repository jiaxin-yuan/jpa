#!/usr/bin/env bash
# Checkpoints -> per-instance predictions -> Tables 1-5 -> comparison with the paper.
set -euo pipefail
cd "$(dirname "$0")/src"
python log_stats.py        # Table 1 raw-log counts             (CPU)
python keys.py             # (case, prefix length) of THP rows  (CPU)
python infer_seq2seq.py    # SuTraN / ED-LSTM inference         (GPU recommended)
python population.py       # evaluation prefix log S, V0-V2     (CPU)
python export_thp.py       # THP-M, THP-B, THP-B + wrapper      (GPU recommended)
python export_seq2seq.py   # SuTraN, ED-LSTM + wrapper          (CPU)
python make_tables.py      # tables/values.json + tab*.tex      (CPU)
cd .. && python verify.py
