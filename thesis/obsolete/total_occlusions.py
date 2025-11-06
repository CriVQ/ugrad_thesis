#!/usr/bin/env python3
import os
import csv

# ─────────────────────────────────────────────────────
# CONFIGURATION: change this if your CSVs live elsewhere
# ─────────────────────────────────────────────────────
RESULTS_DIR = 'results_withOcclusions'   # folder containing *_angles.csv (with an 'occluded' column)

# ─────────────────────────────────────────────────────
# UTILITY: count occlusions in one CSV
# ─────────────────────────────────────────────────────
def count_occlusions(path):
    occl = 0
    total = 0
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get('occluded','').strip() == '1':
                occl += 1
    return occl, total

# ─────────────────────────────────────────────────────
# MAIN: scan all CSVs, write a summary CSV
# ─────────────────────────────────────────────────────
def main():
    # find all .csv files in RESULTS_DIR
    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith('.csv'))
    if not files:
        print(f"No CSVs found in {RESULTS_DIR!r}.")
        return

    summary_path = os.path.join(RESULTS_DIR, 'occlusion_summary_final.csv')
    with open(summary_path, 'w', newline='') as outf:
        writer = csv.writer(outf)
        # header
        writer.writerow(['participant','occluded_frames','total_frames','occlusion_pct'])
        grand_occl = 0
        grand_total = 0

        for fn in files:
            path = os.path.join(RESULTS_DIR, fn)
            occl, tot = count_occlusions(path)
            pct = (occl/tot*100) if tot else 0.0
            participant = os.path.splitext(fn)[0]
            writer.writerow([participant, occl, tot, f"{pct:.1f}"])
            grand_occl += occl
            grand_total += tot

        # overall summary row
        overall_pct = (grand_occl/grand_total*100) if grand_total else 0.0
        writer.writerow([])
        writer.writerow(['TOTAL', grand_occl, grand_total, f"{overall_pct:.1f}"])

    print(f"→ Wrote occlusion summary to {summary_path}")

if __name__=='__main__':
    main()
