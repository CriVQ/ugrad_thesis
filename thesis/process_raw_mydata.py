import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# 1. Cycle extractor for cyclic (flexion) data
def extract_cycles_normalized(series, num_points=101, min_dist=30, min_len=10):
    vals = series.values
    peaks, _ = find_peaks(vals, distance=min_dist)
    cycles = []
    for i in range(len(peaks) - 1):
        seg = vals[peaks[i]:peaks[i+1] + 1]
        if len(seg) < min_len:
            continue
        x_old = np.linspace(0, 100, len(seg))
        f = interp1d(x_old, seg, kind='linear')
        cycles.append(f(np.linspace(0, 100, num_points)))
    return np.array(cycles)

# 2. Normalizer for monotonic (adduction) data
def normalize_monotonic(series, num_points=101):
    y = series.values
    x_old = np.linspace(0, 100, len(y))
    f = interp1d(x_old, y, kind='linear')
    return f(np.linspace(0, 100, num_points))[None, :]  

# 3. Master processing function
def process_deviation_and_cycles(
    folder_path: str,
    columns: list,
    output_folder: str = "processed",
    reference_angle: float = 90.0,
    first_pct: float = 5.0,
    loess_frac: float = 0.1
):
    os.makedirs(output_folder, exist_ok=True)
    percent = np.linspace(0, 100, 101)
    
    for col in columns:
        all_cycles = []
        for fn in os.listdir(folder_path):
            #switch to `.endswith('.xlsx')` and use read_excel
            if not fn.lower().endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(folder_path, fn))
            if col not in df.columns:
                continue

            # 3a) deviation from 90°
            dev = (df[col] - reference_angle).abs().dropna()

            # 3b) pick extraction routine
            if "Abd" in col or "adduction" in col.lower():
                cycles = normalize_monotonic(dev, num_points=101)
            else:
                cycles = extract_cycles_normalized(dev, num_points=101)
            
            if cycles.size:
                all_cycles.append(cycles)

        if not all_cycles:
            print(f" No valid cycles for column {col}")
            continue

        # 4. stack & stats
        A = np.vstack(all_cycles)            # shape = (n_cycles_total, 101)
        mean_dev = A.mean(axis=0)
        sd_dev   = A.std(axis=0)
        loess_dev = lowess(mean_dev, percent, frac=loess_frac, return_sorted=False)

        # 5. save
        out_df = pd.DataFrame({
            "Percent": percent,
            "Mean_Dev90": mean_dev,
            "LOESS_Dev90": loess_dev,
            "SD_Dev90": sd_dev
        })
        out_path = os.path.join(output_folder, f"{col}_processed.xlsx")
        out_df.to_excel(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # Folder containing your raw .csv files
    data_folder = "results"
    # Columns process
    cols_to_do = ["raw_elbow", "raw_shFlex", "raw_shAbd"]
    process_deviation_and_cycles(
        folder_path    = data_folder,
        columns        = cols_to_do,
        output_folder  = "processed",
        reference_angle= 90.0,
        first_pct      = 5.0,
        loess_frac     = 0.1
    )
