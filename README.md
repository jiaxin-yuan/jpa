# JPA: Joint Prediction Accuracy for PPM

Code and artifacts to reproduce Tables 1–5 of *Beyond Marginal Evaluation: Joint Prediction
Accuracy for Predictive Process Monitoring* from the trained checkpoints.

## Run

```bash
pip install -r requirements.txt      # Python 3.12, CUDA 12.4 tested
./run_all.sh                         # ~17 min on one RTX A6000; writes ~5 GB to results/
```

`run_all.sh` runs, in order (one problem per script, all in `src/`):

| step | script | output |
|---|---|---|
| raw-log counts (Table 1) | `log_stats.py` | `results/log_stats.csv` |
| (case, prefix length) of each THP row | `keys.py` | `results/keys/` |
| SuTraN / ED-LSTM inference, val + test | `infer_seq2seq.py` | `results/seq2seq/` |
| evaluation prefix log S, checks V0–V2 | `population.py` | `results/population/` |
| THP-M, THP-B, THP-B + wrapper | `export_thp.py` | `results/predictions/` |
| SuTraN, ED-LSTM + wrapper | `export_seq2seq.py` | `results/predictions/` |
| Tables 1–5 | `make_tables.py` | `tables/values.json`, `tables/tab*.tex` |
| comparison with the paper | `verify.py` | stdout |

Shared modules: `config.py` (paths, logs, ε grid), `jpa.py` (Eq. 2, B2 null, skill Eq. 4,
cMAE Eq. 7), `zilm.py` (Eq. 5 decodes), `wrapper.py` (Eq. 6), `naive.py` (B1),
`stats.py` (exact McNemar, 95% CI, Holm).

## Expected result

```
567/570 cells match the paper exactly; 3 within one instance; 0 mismatches
```

The three cells (Table 3 Sepsis at 42.1 min; Table 5 BPIC12 THP-M−THP-B Δ and lower CI)
each differ by one instance: GPU re-inference changes one THP-B activity on BPIC12
(1 of 34,937) and one SuTraN ε-mode hit on Sepsis (802 vs 803). Everything else, including
S (all 10 logs) and the activity predictions of the other models, is identical to the
paper's run.

## Declared configuration (protocol P1–P3)

- **Decode D (P1).** THP-M: native density (Eq. 5, K = 5). THP-B, SuTraN, ED-LSTM: point
  output; mean / median / ε-mode via the wrapper of Eq. 6, fitted on validation.
  ε-mode is re-decoded at every tolerance of the grid.
- **Evaluation prefix log S (P2).** Strict temporal split (THP partition), then prefix-level
  intersection keyed by (case, prefix length) with checks V0–V2 (`population.py`).
- **Criterion (P3).** Absolute JPA_ε, 25 log-spaced tolerances from 86.4 ms to 5 d,
  ε* = 193 s (grid index 12).
- **Scoring decode in cross-model comparisons.** Table 2 (JPA, cMAE), Table 3 (flips),
  Table 5 and the B1 tests score THP-B by its point prediction; the other models use
  ε-mode for JPA and median for cMAE. Table 3 spreads and Table 4 density columns use the
  THP-B wrapper.

## Contents

| path | content | origin |
|---|---|---|
| `data/raw/` | 10 raw XES logs | UQ4PPM benchmark logs |
| `data/thp/` | THP train/val/test splits (fold 0) | strict temporal split |
| `data/seq2seq/` | SuTraN-benchmark val/test tensors + metadata on the same split | |
| `checkpoints/thp/` | THP-M (`mixture`) and THP-B (`baseline_act`), best val next-event MAE | |
| `checkpoints/seq2seq/` | SuTraN, ED-LSTM, epoch with best val remaining-time MAE (`*_backup_results.csv` = training log) | |
| `models/thp/` | THP encoder and data loader (unchanged) | THP_for_PPM |
| `models/seq2seq/` | SuTraN and ED-LSTM (unchanged, see its LICENSE) | SuTraN benchmark |
| `reference/paper_tables.json` | Tables 1–5 as printed in the paper | |

Training code is not included: every model was trained once (one split, one seed), and the
tables are reproduced from those checkpoints.
