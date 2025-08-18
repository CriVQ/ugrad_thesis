#!/usr/bin/env python3
import os
import csv
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────
RECS_DIR    = 'recordings'
RESULTS_DIR = 'resultsRaw'
MODEL_PATH  = 'C:/Users/ADAM/Downloads/metrabs/'  # adjust to your path

# ─────────────────────────────────────────────────────
# LOAD MODEL & INDEX JOINTS
# ─────────────────────────────────────────────────────
model = hub.load(MODEL_PATH)
joint_names = model.per_skeleton_joint_names['smpl+head_30'] \
              .numpy().astype(str).tolist()
idx = {
    'rsho': joint_names.index('rsho_smpl'),
    'lsho': joint_names.index('lsho_smpl'),
    'relb': joint_names.index('relb_smpl'),
    'rwri': joint_names.index('rwri_smpl'),
    'neck': joint_names.index('neck_smpl'),
}

# ─────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────
def calculate_angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6:
        return 0.0
    c = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def midpoint(a, b):
    return (a + b) * 0.5

def sample_raws(p3):
    """Compute raw elbow, shoulder flexion & abduction."""
    sh   = p3[idx['rsho']]
    lsh  = p3[idx['lsho']]
    el   = p3[idx['relb']]
    wr   = p3[idx['rwri']]
    neck = p3[idx['neck']]

    # elbow
    upper = el - sh
    lower = wr - el
    raw_e = calculate_angle(upper, lower)

    # shoulder axes
    sho_mid  = midpoint(sh, lsh)
    up_axis  = neck - sho_mid
    up_axis /= np.linalg.norm(up_axis) + 1e-9

    lat_axis = (lsh - sh)
    lat_axis /= np.linalg.norm(lat_axis) + 1e-9

    # sag plane
    sag = upper - np.dot(upper, lat_axis)*lat_axis
    raw_sf = calculate_angle(sag, up_axis)

    # cor plane
    fwd = np.cross(up_axis, lat_axis)
    fwd /= np.linalg.norm(fwd) + 1e-9
    cor = upper - np.dot(upper, fwd)*fwd
    raw_sa = calculate_angle(cor, up_axis)

    return raw_e, raw_sf, raw_sa

# ─────────────────────────────────────────────────────
# MAIN LOOP (headless, raw + occlusion)
# ─────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)

for pid in sorted(os.listdir(RECS_DIR)):
    subj_dir   = os.path.join(RECS_DIR, pid)
    session_mp4 = os.path.join(subj_dir, 'session.mp4')
    if not os.path.isdir(subj_dir) or not os.path.isfile(session_mp4):
        continue

    out_csv = os.path.join(RESULTS_DIR, f'{pid}_raw_angles.csv')
    print(f"→ Processing subject {pid}")

    cap = cv2.VideoCapture(session_mp4)
    if not cap.isOpened():
        print(f"   ✗ Cannot open {session_mp4}")
        continue

    total_frames = 0
    occluded     = 0

    with open(out_csv, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            'frame', 'time_s',
            'raw_elbow', 'raw_shFlex', 'raw_shAbd',
            'occluded'
        ])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_frames += 1
            # accurate timestamp in seconds
            time_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # pose detect
            tfimg = tf.image.convert_image_dtype(frame, tf.uint8)
            pred  = model.detect_poses(tfimg, skeleton='smpl+head_30')
            p3s   = pred['poses3d'].numpy()

            if p3s.shape[0] == 0:
                # occluded
                occluded += 1
                writer.writerow([
                    frame_idx,
                    f"{time_s:.3f}",
                    '', '', '',
                    1
                ])
            else:
                # valid detection
                p3s[0][...,1], p3s[0][...,2] = p3s[0][...,2], -p3s[0][...,1]
                raw_e, raw_sf, raw_sa = sample_raws(p3s[0])
                writer.writerow([
                    frame_idx,
                    f"{time_s:.3f}",
                    f"{raw_e:.3f}",
                    f"{raw_sf:.3f}",
                    f"{raw_sa:.3f}",
                    0
                ])

            frame_idx += 1

    cap.release()
    rate = occluded/total_frames*100 if total_frames>0 else 0.0
    print(f"   ✓ Done: {occluded}/{total_frames} occluded ({rate:.1f}%)")
    print(f"   CSV → {out_csv}")

print("All subjects complete.")
