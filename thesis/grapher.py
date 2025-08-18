import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

# ─── CONFIG (EDIT THESE) ───────────────────────────────────────────────────
VR_XLSX              = "all_participant_metricsFinal.xlsx"   # VR raw: one sheet with BOTH metrics
VR_SHEET             = 0                                      # index or name of VR sheet
KONTSON_XLSX         = "KontsonDataRaw.xlsx"                  # Kontson reference

# Use sheet indexes or exact names (adjust if different in your file)
KONTSON_FLEX_SHEET   = 1                                      # e.g., "Flexion"
KONTSON_ABD_SHEET    = 2                                      # e.g., "Adduction"

NUM_POINTS           = 101                                    # resample to 0..100% (101 bins)
ALIGN_START          = True                                   # baseline-shift VR to match start

# >>> VR ABDUCTION TRIM (only VR is trimmed; Kontson stays full) <<<
VR_ABD_START_FRAC    = 0.0   # 10% cut at start (0.00–1.00)
VR_ABD_END_FRAC      = 1.0  # end of window (0.00–1.00); e.g., 0.60 to stop at 60%

# Robust smoothing settings
LOESS_FRAC_TRIALS    = 0.30   # per-trial LOESS span
LOESS_FRAC_GROUP     = 0.25   # group LOESS span
POST_MOVING_AVG_K    = 3      # moving-average window (1 disables)
AGGREGATION_METHOD   = "median"  # "median" (robust) or winsorized mean
# ───────────────────────────────────────────────────────────────────────────


# -------------------------- helpers ---------------------------------------

def load_sheet_numeric(xlsx, sheet):
    df = pd.read_excel(xlsx, sheet_name=sheet, header=0)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        df = df.apply(pd.to_numeric, errors="coerce")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[num_cols].dropna(how="all").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No numeric data in sheet '{sheet}' of {xlsx}")
    return df

def split_vr_flex_abd(df):
    """Split VR columns into Flexion vs Abduction by keywords; fallback even/odd."""
    cols = list(df.columns)
    flex_cols = [c for c in cols if re.search(r'flex', str(c), re.I)]
    abd_cols  = [c for c in cols if re.search(r'(abd|add|adduction)', str(c), re.I)]
    if not flex_cols or not abd_cols:
        # fallback: even = Flexion, odd = Abduction
        flex_cols = cols[::2]
        abd_cols  = cols[1::2]
    if not flex_cols or not abd_cols:
        raise ValueError("Could not split VR sheet into Flexion/Abduction groups.")
    return df[flex_cols], df[abd_cols]

def normalize_trials(df, num_points=101):
    """Interpolate each column to num_points over 0..100%."""
    percent = np.linspace(0, 100, num_points)
    trials = []
    for c in df.columns:
        y = pd.to_numeric(df[c], errors="coerce").dropna().values
        if len(y) < 2:
            continue
        f = interp1d(np.linspace(0, 100, len(y)), y, kind="linear")
        trials.append(f(percent))
    if not trials:
        raise ValueError("No usable time series found after normalization.")
    return np.vstack(trials), percent

def loess_smooth(y, x, frac=0.25, it=1):
    """Robust LOWESS with wider span for stability."""
    return lowess(y, x, frac=frac, it=it, return_sorted=False)

def per_trial_loess(trials, percent, frac=0.30, it=1):
    sm = np.empty_like(trials)
    for i in range(trials.shape[0]):
        sm[i] = loess_smooth(trials[i], percent, frac=frac, it=it)
    return sm

def robust_aggregate(trials, how="median"):
    if how == "median":
        return np.nanmedian(trials, axis=0)
    else:
        lo = np.percentile(trials, 5, axis=0)
        hi = np.percentile(trials, 95, axis=0)
        tw = np.clip(trials, lo, hi)
        return np.nanmean(tw, axis=0)

def moving_avg(y, k=3):
    if k <= 1:
        return y
    k = int(k)
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(ypad, kernel, mode="valid")

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b)**2)))

def mad(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))

def load_kontson_metric(xlsx, sheet, num_points=101):
    df = load_sheet_numeric(xlsx, sheet)
    s_cols = [c for c in [f"S{i}" for i in range(1,20)] if c in df.columns]
    df = df[s_cols] if s_cols else df
    return normalize_trials(df, num_points)

def choose_orientation_and_align(vr_curve, vr_pct, kont_curve, kont_pct, align=True):
    """
    Align AFTER trimming VR: interpolate Kontson onto VR's (possibly trimmed) grid,
    then test normal vs flipped VR and pick the lower-RMSE orientation.
    Returns: best_vr_on_vrgrid, shift_used, flipped_bool, rmse_best, mad_best
    """
    kont_on_vr = np.interp(vr_pct, kont_pct, kont_curve)

    def orient_stats(v):
        v_aligned = v.copy()
        shift = 0.0
        if align:
            shift = kont_on_vr[0] - v_aligned[0]  # match at start of trimmed VR window
            v_aligned = v_aligned + shift
        return v_aligned, shift, rmse(v_aligned, kont_on_vr), mad(v_aligned, kont_on_vr)

    # Test both orientations
    vA, shiftA, rmseA, madA = orient_stats(vr_curve)    # as-is
    vB, shiftB, rmseB, madB = orient_stats(-vr_curve)   # flipped

    if rmseB < rmseA:
        return vB, shiftB, True, rmseB, madB
    else:
        return vA, shiftA, False, rmseA, madA


# -------------------- dedicated run functions (isolated) -------------------

def run_flex_full(vr_df_metric, kont_sheet, ylabel, title):
    # VR (NO TRIM)
    vr_trials, vr_pct = normalize_trials(vr_df_metric, NUM_POINTS)
    vr_trials_sm = per_trial_loess(vr_trials, vr_pct, frac=LOESS_FRAC_TRIALS, it=1)
    vr_group     = robust_aggregate(vr_trials_sm, how=AGGREGATION_METHOD)
    vr_sm        = loess_smooth(vr_group, vr_pct, frac=LOESS_FRAC_GROUP, it=1)
    vr_sm        = moving_avg(vr_sm, k=POST_MOVING_AVG_K)

    # Kontson (NO TRIM)
    kont_trials, kont_pct = load_kontson_metric(KONTSON_XLSX, kont_sheet, NUM_POINTS)
    kont_trials_sm = per_trial_loess(kont_trials, kont_pct, frac=LOESS_FRAC_TRIALS, it=1)
    kont_group     = robust_aggregate(kont_trials_sm, how=AGGREGATION_METHOD)
    kont_sm        = loess_smooth(kont_group, kont_pct, frac=LOESS_FRAC_GROUP, it=1)
    kont_sm        = moving_avg(kont_sm, k=POST_MOVING_AVG_K)

    # Auto orientation + (optional) baseline align; compare on VR grid
    vr_best, shift, flipped, _rmse, _mad = choose_orientation_and_align(
        vr_curve=vr_sm, vr_pct=vr_pct, kont_curve=kont_sm, kont_pct=kont_pct, align=ALIGN_START
    )

    print(f"[Flexion] VR_pts={len(vr_pct)}, Kont_pts={len(kont_pct)}, flipped={flipped}, "
          f"shift={shift:.2f}°, RMSE={_rmse:.2f}°, MAD={_mad:.2f}°")

    plt.figure(figsize=(8,5))
    plt.plot(vr_pct,   vr_best, lw=2, label=f"VR-BBT Flexion{' (flipped)' if flipped else ''}{', shift %.2f°' % shift if ALIGN_START else ''} (full)")
    plt.plot(kont_pct, kont_sm, lw=2, linestyle="--", label="Kontson Flexion (full)")
    plt.xlabel("Percent of Movement Cycle (%)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


def run_abd_vr_trim(vr_df_metric, kont_sheet, ylabel, title,
                    vr_start_frac=0.10, vr_end_frac=1.0):
    # --- VR (TRIM FIRST) ---
    vr_trials, _ = normalize_trials(vr_df_metric, NUM_POINTS)
    i0 = int(round(vr_start_frac * NUM_POINTS))
    i1 = int(round(vr_end_frac   * NUM_POINTS))
    i0 = max(0, min(NUM_POINTS-2, i0))
    i1 = max(i0+1, min(NUM_POINTS, i1))
    vr_trials = vr_trials[:, i0:i1]                     # VR-only cut first
    vr_pct    = np.linspace(0, 100, vr_trials.shape[1]) # reindex VR window to 0–100

    vr_trials_sm = per_trial_loess(vr_trials, vr_pct, frac=LOESS_FRAC_TRIALS, it=1)
    vr_group     = robust_aggregate(vr_trials_sm, how=AGGREGATION_METHOD)
    vr_sm        = loess_smooth(vr_group, vr_pct, frac=LOESS_FRAC_GROUP, it=1)
    vr_sm        = moving_avg(vr_sm, k=POST_MOVING_AVG_K)

    # --- Kontson (NO TRIM for plotting) ---
    kont_trials, kont_pct_full = load_kontson_metric(KONTSON_XLSX, kont_sheet, NUM_POINTS)
    kont_trials_sm = per_trial_loess(kont_trials, kont_pct_full, frac=LOESS_FRAC_TRIALS, it=1)
    kont_group     = robust_aggregate(kont_trials_sm, how=AGGREGATION_METHOD)
    kont_sm_full   = loess_smooth(kont_group, kont_pct_full, frac=LOESS_FRAC_GROUP, it=1)
    kont_sm_full   = moving_avg(kont_sm_full, k=POST_MOVING_AVG_K)

    # Align AFTER trim using Kontson evaluated on VR's (trimmed) grid
    vr_best, shift, flipped, _rmse, _mad = choose_orientation_and_align(
        vr_curve=vr_sm, vr_pct=vr_pct,
        kont_curve=kont_sm_full, kont_pct=kont_pct_full,
        align=ALIGN_START
    )

    print(f"[Adduction] VR window={int(vr_start_frac*100)}–{int(vr_end_frac*100)}% "
          f"(VR_pts={len(vr_pct)}), Kont_pts(full)={len(kont_pct_full)}, "
          f"flipped={flipped}, shift={shift:.2f}°, RMSE={_rmse:.2f}°, MAD={_mad:.2f}°")

    # Plot: VR (trimmed window relabeled 0–100), Kontson full for context
    plt.figure(figsize=(8,5))
    win_note = f" (VR {int(vr_start_frac*100)}–{int(vr_end_frac*100)}%)"
    plt.plot(vr_pct,         vr_best,      lw=2,              label=f"VR-BBT Adduction{' (flipped)' if flipped else ''}{', shift %.2f°' % shift if ALIGN_START else ''}{win_note}")
    plt.plot(kont_pct_full,  kont_sm_full, lw=2, linestyle="--", label="Kontson Adduction (full)")
    plt.xlabel("Percent of Movement Cycle (%)"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()


# ------------------------------- RUN --------------------------------------
vr_all = load_sheet_numeric(VR_XLSX, VR_SHEET)
vr_flex_df, vr_abd_df = split_vr_flex_abd(vr_all)

# Flexion — full cycle only
run_flex_full(
    vr_df_metric=vr_flex_df,
    kont_sheet=KONTSON_FLEX_SHEET,
    ylabel="Shoulder Flexion Angle (°)",
    title="Flexion: VR-BBT vs. Kontson (Robust LOESS, full; post-trim alignment)"
)

# Abduction — VR-only trim (controls at top: VR_ABD_START_FRAC / VR_ABD_END_FRAC)
run_abd_vr_trim(
    vr_df_metric=vr_abd_df,
    kont_sheet=KONTSON_ABD_SHEET,
    ylabel="Shoulder Adduction Angle (°)",
    title="Adduction: VR-BBT vs. Kontson (Robust LOESS; VR-only trim; post-trim alignment)",
    vr_start_frac=VR_ABD_START_FRAC,
    vr_end_frac=VR_ABD_END_FRAC
)
