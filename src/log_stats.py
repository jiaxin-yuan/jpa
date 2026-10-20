"""Count cases, events and activities (concept:name) on each raw XES log -> results/log_stats.csv."""
import os
import warnings

import pandas as pd
import pm4py

import config as C

warnings.filterwarnings("ignore")


def main():
    """Write one row per log, in the order of config.LOGS."""
    rows = []
    for ds in C.LOGS:
        log = pm4py.read_xes(os.path.join(C.DATA, "raw", f"{ds[3:]}.xes"))
        if not isinstance(log, pd.DataFrame):
            log = pm4py.convert_to_dataframe(log)
        rows.append({"log": C.SHORT[ds], "cases": log["case:concept:name"].nunique(),
                     "events": len(log), "activities": log["concept:name"].nunique()})
        print(rows[-1])
    os.makedirs(C.RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "log_stats.csv"), index=False)


if __name__ == "__main__":
    main()
