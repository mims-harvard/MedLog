"""Synthetic test data generation for fairness analysis."""

from typing import Optional

import numpy as np
import pandas as pd


def make_dummy_assembled(
    n_patients: int = 80,
    icu_days_range: tuple = (1, 10),
    circ_event_prob: float = 0.25,
    resp_event_prob: float = 0.20,
    seed: int = 42,
    out_path: Optional[str] = "dummy_assembled.parquet",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate a synthetic assembled.parquet with the same schema used by
    paper_figure_4_v2_fairness.ipynb and fairness.ipynb.

    Columns produced
    ----------------
    datetime                       : UTC datetime (5-min grid)
    patient_id                     : int64
    sex                            : 'M' | 'F'
    age                            : int (years)
    circ_raw_prediction            : float [0, 1]  — continuous model score
    resp_raw_prediction            : float [0, 1]
    Dynamic_CircFailure_8Hours     : int  {0, 1}   — binary label
    Dynamic_RespFailure_mid_8Hours : int  {0, 1}
    circ_alarm                     : bool
    resp_alarm                     : bool
    """
    rng = np.random.default_rng(seed)
    STEP = pd.Timedelta("5min")
    START = pd.Timestamp("2024-01-01", tz="UTC")
    CIRC_THRESHOLD = 0.225
    RESP_THRESHOLD = 0.146

    rows = []
    for pid in range(1, n_patients + 1):
        n_days = rng.integers(*icu_days_range, endpoint=True)
        n_steps = n_days * 24 * 12   # 12 five-minute steps per hour

        sex = rng.choice(["M", "F"])
        age = int(rng.integers(35, 85))
        t0 = START + pid * pd.Timedelta("3days")   # stagger admission times

        has_circ = rng.random() < circ_event_prob
        has_resp = rng.random() < resp_event_prob

        def _event_window(n, present):
            mask = np.zeros(n, dtype=int)
            if not present:
                return mask
            dur = rng.integers(12, min(12 * 24, n // 2), endpoint=True)  # 1h–12h
            start = rng.integers(0, max(1, n - dur))
            mask[start: start + dur] = 1
            return mask

        circ_label = _event_window(n_steps, has_circ)
        resp_label = _event_window(n_steps, has_resp)

        def _scores(label, base_pos=0.45, base_neg=0.08, noise=0.12):
            base = np.where(label, base_pos, base_neg)
            s = base + rng.normal(0, noise, len(label))
            return np.clip(s, 0, 1)

        circ_score = _scores(circ_label)
        resp_score = _scores(resp_label)

        for i in range(n_steps):
            rows.append({
                "datetime": t0 + i * STEP,
                "patient_id": pid,
                "sex": sex,
                "age": age,
                "circ_raw_prediction": circ_score[i],
                "resp_raw_prediction": resp_score[i],
                "Dynamic_CircFailure_8Hours": circ_label[i],
                "Dynamic_RespFailure_mid_8Hours": resp_label[i],
                "circ_alarm": bool(circ_score[i] >= CIRC_THRESHOLD),
                "resp_alarm": bool(resp_score[i] >= RESP_THRESHOLD),
            })

    df = pd.DataFrame(rows)
    df["patient_id"] = df["patient_id"].astype(np.int64)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    if out_path is not None:
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df):,} rows ({df['patient_id'].nunique()} patients) -> {out_path}")

    if verbose:
        print(f"Patients per sex: {df.groupby('sex')['patient_id'].nunique().to_dict()}")
        print(f"Circ events: {df['Dynamic_CircFailure_8Hours'].sum():,} / {len(df):,} "
              f"({df['Dynamic_CircFailure_8Hours'].mean():.1%})")
        print(f"Resp events: {df['Dynamic_RespFailure_mid_8Hours'].sum():,} / {len(df):,} "
              f"({df['Dynamic_RespFailure_mid_8Hours'].mean():.1%})")

    return df
