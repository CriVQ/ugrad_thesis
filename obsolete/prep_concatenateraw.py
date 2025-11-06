import pandas as pd
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_FOLDER = "results"                 # folder containing trial CSVs
METRICS     = ["true_shFlex", "true_shAbd", "raw_elbow"]
OUTPUT_XLSX = "all_participant_metricsFinal1.xlsx"
# ────────────────────────────────────────────────────────────────────────────

# 1) Gather each metric into a dict of lists (one key per participant)
metric_dicts = {m: {} for m in METRICS}

for fn in sorted(os.listdir(DATA_FOLDER)):
    if not fn.lower().endswith(".csv"):
        continue
    subj_name = os.path.splitext(fn)[0]
    df = pd.read_csv(os.path.join(DATA_FOLDER, fn))
    for m in METRICS:
        if m in df.columns:
            metric_dicts[m][subj_name] = df[m].dropna().reset_index(drop=True)

# 2) Build a DataFrame per metric, aligning by row index (missing values become NaN)
sheets = {}
for m, d in metric_dicts.items():
    sheets[m] = pd.DataFrame(d)

# 3) Write all three to separate sheets in one Excel file
with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    sheets["true_shFlex"].to_excel(writer, sheet_name="Flexion", index=False)
    sheets["true_shAbd"].to_excel(writer, sheet_name="Abduction", index=False)
    sheets["raw_elbow"].to_excel(writer, sheet_name="Elbow", index=False)

print(f"Exported participant metrics to '{OUTPUT_XLSX}' with sheets: Flexion, Abduction, Elbow")
