import os
import argparse
import json
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ANGLE_COLUMNS = ["raw_elbow_deg", "raw_shFlex_deg", "raw_shAbd_deg"]

# Segment Loading 
def load_segments(segments_csv_path: str):
  
    if not os.path.isfile(segments_csv_path):
        return []
    if os.path.getsize(segments_csv_path) == 0:
        print(f"[segments] Empty file, skipping: {segments_csv_path}")
        return []
    try:
        df = pd.read_csv(segments_csv_path)
    except EmptyDataError:
        print(f"[segments] EmptyDataError: {segments_csv_path}")
        return []
    except Exception as e:
        print(f"[segments] Read error {segments_csv_path}: {e}")
        return []
    if df.empty:
        return []
    df.columns = [c.strip() for c in df.columns]
    start_col = 'start_frame'
    end_col = 'end_frame'
    if start_col not in df.columns or end_col not in df.columns:
        lowers = {c.lower(): c for c in df.columns}
        if 'start' in lowers and 'end' in lowers:
            start_col = lowers['start']
            end_col = lowers['end']
        else:
            print(f"[segments] Missing start_frame/end_frame columns in {segments_csv_path}")
            return []
    segments = []
    for row in df.itertuples():
        try:
            s = int(getattr(row, start_col))
            e = int(getattr(row, end_col))
            segments.append((min(s, e), max(s, e)))
        except Exception:
            continue
    return segments

# Interpolation Helpers 
def interpolate_cycle(frames: np.ndarray,
                      values: np.ndarray,
                      start_frame: int,
                      end_frame: int,
                      num_points: int = 101):

    finite_mask = np.isfinite(values)
    if finite_mask.sum() < 2:
        return np.full(num_points, np.nan)
    frames_finite = frames[finite_mask]
    values_finite = values[finite_mask]
    target_frames = np.linspace(start_frame, end_frame, num_points)
    try:
        return np.interp(target_frames, frames_finite, values_finite)
    except Exception:
        return np.full(num_points, np.nan)

# -------------------- Participant Processing --------------------
def process_participant(participant: str,
                        angles_csv: str,
                        segments_csv: str,
                        num_points: int = 101,
                        angle_columns = ANGLE_COLUMNS):

    if not os.path.isfile(angles_csv):
        print(f"[WARN] Missing angles file for {participant}: {angles_csv}")
        return {a: [] for a in angle_columns}, 0, 0

    segments = load_segments(segments_csv)
    if not segments:
        print(f"[INFO] No segments (or empty) for {participant}.")
        return {a: [] for a in angle_columns}, 0, 0

    df = pd.read_csv(angles_csv)
    if 'frame' not in df.columns:
        raise ValueError(f"Angles CSV for {participant} missing 'frame' column.")
    df = df.sort_values('frame').reset_index(drop=True)

    for col in angle_columns:
        if col not in df.columns:
            df[col] = np.nan

    frame_arr = df['frame'].to_numpy()
    cycles_by_angle = {col: [] for col in angle_columns}
    skipped = 0
    used = 0

    for (s, e) in segments:
        seg_mask = (frame_arr >= s) & (frame_arr <= e)
        if not seg_mask.any():
            skipped += 1
            continue
        seg_frames = frame_arr[seg_mask]
        if len(np.unique(seg_frames)) < 2:
            skipped += 1
            continue

        for col in angle_columns:
            seg_vals = df.loc[seg_mask, col].to_numpy(dtype=float)
            norm_vals = interpolate_cycle(seg_frames, seg_vals, s, e, num_points=num_points)
            cycles_by_angle[col].append(norm_vals)
        used += 1

    return cycles_by_angle, skipped, used

#  Workbook Assembly Helpers 
def build_participant_cycles_only_df(cycles_by_angle: dict,
                                     num_points: int = 101,
                                     angle_columns = ANGLE_COLUMNS):
   
    percent = np.linspace(0, 100, num_points)
    col_data = {"percent": percent}
    for angle in angle_columns:
        cycles = cycles_by_angle[angle]
        if cycles:
            arr = np.vstack(cycles)  # (num_cycles, num_points)
            for i, cyc in enumerate(arr, start=1):
                col_data[f"{angle}_cycle{i}"] = cyc
    return pd.DataFrame(col_data)

# - Main Script 
def main():
    ap = argparse.ArgumentParser(description="Normalize cycles (0-100%), compute participant & angle-grouped global means.")
    ap.add_argument("--indir", required=True)

