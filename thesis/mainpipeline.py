import os
import cv2
import json
import csv
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import numpy.polynomial.polynomial as poly
from cv2 import VideoWriter_fourcc
from moviepy import ImageSequenceClip
from moviepy import VideoFileClip
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob
import subprocess
from scipy.spatial.transform import Rotation as R
import argparse
import subprocess


RECS_DIR    = 'recordings'
CAL_DIR     = 'calibrationsExperimental'
RESULTS_DIR = 'results_withOcclusions'
TRUE_MAX    = 150.0
MODEL_PATH  = 'C:/Users/ADAM/Downloads/metrabs/'

model = hub.load(MODEL_PATH)
jn    = model.per_skeleton_joint_names['smpl+head_30'].numpy().astype(str).tolist()
idx   = { name: jn.index(name) for name in (
    'rsho_smpl','relb_smpl','rwri_smpl',
    'lsho_smpl','neck_smpl'
)}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--start',
    help="only begin processing at this subject folder name",
    default=None
)
args = parser.parse_args()

def calculate_angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6:
        return 0.0
    c = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def midpoint(a, b):
    return (a + b) * 0.5

def fit_scale(raw_vals, true_vals):
    coefs = poly.Polynomial.fit(raw_vals, true_vals, deg=1).convert().coef
    return coefs[0], coefs[1]  # intercept, slope


def sample_raws(p3):
    # Joint positions
    sh   = p3[idx['rsho_smpl']]
    lsh = p3[idx['lsho_smpl']]
    el   = p3[idx['relb_smpl']]
    wr   = p3[idx['rwri_smpl']]
    neck = p3[idx['neck_smpl']]

    # Upper & lower arm
    upper = el - sh
    lower = wr - el
    raw_e = calculate_angle(upper, lower)

    # Build local axes using neck
    sho_mid  = midpoint(sh, lsh)
    up_axis  = neck - sho_mid
    up_axis /= (np.linalg.norm(up_axis) + 1e-9)

    lat_axis = (p3[idx['lsho_smpl']] - sh)
    lat_axis /= np.linalg.norm(lat_axis) + 1e-9

    fwd_axis = np.cross(up_axis, lat_axis)
    fwd_axis /= np.linalg.norm(fwd_axis) + 1e-9

    # Shoulder flexion: project onto plane orthogonal to lat_axis (sagittal)
    sag_plane = upper - np.dot(upper, lat_axis)*lat_axis
    raw_sf = calculate_angle(sag_plane, up_axis)

    # Shoulder abduction: project onto plane orthogonal to fwd_axis (coronal)
    cor_plane = upper - np.dot(upper, fwd_axis)*fwd_axis
    raw_sa = calculate_angle(cor_plane, up_axis)

    return raw_e, raw_sf, raw_sa

PROMPTS = [
    ("Elbow 0° (straight)",      'elbow0'),
    ("Elbow 90° (L-shape)",      'elbow90'),
    ("Elbow max flex",           'elbowMax'),
    ("Shoulder 0° (arm down)",   'sh0'),
    ("Shoulder 90° flex (front)",'shFlex90'),
    ("Shoulder max flex",        'shFlexMax'),
    ("Shoulder 90° abd (side)",  'shAbd90'),
    ("Shoulder max abd",         'shAbdMax'),
]

def calibrate_from_video(calib_vid, calib_json):
    # 1) Load existing calibration if present
    if os.path.exists(calib_json):
        print(f"→ Loaded existing calibration: {calib_json}")
        return json.load(open(calib_json, 'r'))

    cap = cv2.VideoCapture(calib_vid)
    cv2.namedWindow('Calibrate', cv2.WINDOW_NORMAL)
    data = {key: [] for _, key in PROMPTS}

    # 2) Initial 15‑second “get ready” freeze
    ok, frame0 = cap.read()
    if not ok:
        print("Cannot read first frame of calibration video.")
        cap.release()
        cv2.destroyAllWindows()
        return None

    print("📷 Get ready… Calibration starts in 15 s")
    start = time.time()
    while time.time() - start < 15:
        disp = frame0.copy()
        secs = int(15 - (time.time() - start))
        cv2.putText(disp, f"Starting in {secs}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Calibrate', disp)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    # 3) Prompt loop with live video, and immediate 60‑frame capture
    for prompt, key in PROMPTS:
        print(f"\n Hold pose: {prompt} — press ‘C’ to capture")
        # live video until C is pressed
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.putText(frame, f"{prompt} (press C)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Calibrate', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        # sample 60 frames immediately
        for _ in range(60):
            ok, frame = cap.read()
            if not ok:
                continue
            tfimg = tf.image.convert_image_dtype(frame, tf.uint8)
            pred  = model.detect_poses(tfimg, skeleton='smpl+head_30')
            p3s   = pred['poses3d'].numpy()
            if len(p3s) == 0:
                continue
            p3 = p3s[0]; p3[...,1], p3[...,2] = p3[...,2], -p3[...,1]
            raw_e, raw_sf, raw_sa = sample_raws(p3)
            if key.startswith('elbow'):
                data[key].append(raw_e)
            elif key.startswith('shFlex'):
                data[key].append(raw_sf)
            elif key.startswith('shAbd'):
                data[key].append(raw_sa)
            elif key == 'sh0':
                data[key].append(raw_sf)

    cap.release()
    cv2.destroyAllWindows()
    
    # Fit linear mappings (0°, 90°, 150°)
    raw_e   = [np.mean(data[k]) for k in ('elbow0','elbow90','elbowMax')]
    true_e  = [0.0,90.0,TRUE_MAX]
    off_e, s_e = fit_scale(raw_e, true_e)

    raw_sf  = [np.mean(data[k]) for k in ('sh0','shFlex90','shFlexMax')]
    true_sf = [0.0,90.0,TRUE_MAX]
    off_sf, s_sf = fit_scale(raw_sf, true_sf)

    raw_sa  = [np.mean(data[k]) for k in ('sh0','shAbd90','shAbdMax')]
    true_sa = [0.0,90.0,TRUE_MAX]
    off_sa, s_sa = fit_scale(raw_sa, true_sa)



    calib = {
      'base_elbow': off_e,      'scale_elbow': s_e,
      'base_sh0':   off_sf,     'scale_shflex': s_sf,
                             'scale_shabd':  s_sa
    }
    os.makedirs(os.path.dirname(calib_json), exist_ok=True)
    with open(calib_json,'w') as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration → {calib_json}")
    return calib


def track_with_calibration(sess_vid, calib, out_csv, out_vid):
    cap        = cv2.VideoCapture(sess_vid)
    fps        = 30.0
    w, h       = int(cap.get(3)), int(cap.get(4))
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    writer_vid = cv2.VideoWriter(out_vid, fourcc, fps, (w,h))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    csvf   = open(out_csv,'w',newline='')
    writer = csv.writer(csvf)
    # <-- add the 'occluded' column -->
    writer.writerow([
        'frame','time_s',
        'raw_elbow','true_elbow',
        'raw_shFlex','true_shFlex',
        'raw_shAbd','true_shAbd',
        'q_sho_x','q_sho_y','q_sho_z','q_sho_w',
        'occluded'
    ])

    total_frames    = 0
    occluded_frames = 0
    frame = 0
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    print(f" Processing session {sess_vid}")

    while True:
        ok, img = cap.read()
        if not ok:
            break
        total_frames += 1
        time_s = frame / fps

        tfimg = tf.image.convert_image_dtype(img, tf.uint8)
        pred  = model.detect_poses(tfimg, skeleton='smpl+head_30')
        p3s   = pred['poses3d'].numpy()
        p2s   = pred['poses2d'].numpy() if pred['poses2d'] is not None else None

        if len(p3s)==0 or p2s is None:
            occluded_frames += 1
            # write blank angles + flag=1
            writer.writerow([
                frame, f"{time_s:.3f}",
                '', '',  # raw_elbow, true_elbow
                '', '',  # raw_shFlex, true_shFlex
                '', '',  # raw_shAbd, true_shAbd
                '', '', '', '',  # q_sho_x, y, z, w
                1
            ])
        else:
            # your existing angle + quaternion code:
            p3 = p3s[0]; p3[...,1],p3[...,2] = p3[...,2],-p3[...,1]
            raw_e, raw_sf, raw_sa = sample_raws(p3)

            el = np.clip((raw_e - calib['base_elbow'])*calib['scale_elbow'],   0, TRUE_MAX)
            sf = np.clip((raw_sf- calib['base_sh0']  )*calib['scale_shflex'],  0, TRUE_MAX)
            sa = np.clip((raw_sa- calib['base_sh0']  )*calib['scale_shabd'],   0, TRUE_MAX)

            sho_mid  = midpoint(p3[idx['rsho_smpl']], p3[idx['lsho_smpl']])
            neck     = p3[idx['neck_smpl']]
            up_axis  = neck - sho_mid
            up_axis /= (np.linalg.norm(up_axis) + 1e-9)
            lat_axis = p3[idx['lsho_smpl']] - p3[idx['rsho_smpl']]
            lat_axis /= (np.linalg.norm(lat_axis) + 1e-9)
            fwd_axis = np.cross(up_axis, lat_axis)
            fwd_axis /= (np.linalg.norm(fwd_axis) + 1e-9)

            # build quaternion for shoulder frame
            R_sho = np.column_stack((lat_axis, up_axis, fwd_axis))
            q_sho = R.from_matrix(R_sho).as_quat()   # [x, y, z, w]

            # draw as before...
            p2 = p2s[0]
            for a,b in ((idx['rsho_smpl'],idx['relb_smpl']),
                        (idx['relb_smpl'],idx['rwri_smpl'])):
                cv2.line(img,
                         tuple(p2[a].astype(int)),
                         tuple(p2[b].astype(int)),
                         (0,255,0),2)
            for j in ('rsho_smpl','relb_smpl','rwri_smpl'):
                cv2.circle(img,
                           tuple(p2[idx[j]].astype(int)),
                           5, (0,0,255),-1)
            cv2.putText(img, f"E:{el:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(img, f"SF:{sf:.1f}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(img, f"SA:{sa:.1f}", (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            # write full row + flag=0
            writer.writerow([
                frame, f"{time_s:.3f}",
                f"{raw_e:.3f}", f"{el:.3f}",
                f"{raw_sf:.3f}", f"{sf:.3f}",
                f"{raw_sa:.3f}", f"{sa:.3f}",
                f"{q_sho[0]:.6f}", f"{q_sho[1]:.6f}",
                f"{q_sho[2]:.6f}", f"{q_sho[3]:.6f}",
                0
            ])

        writer_vid.write(img)
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        frame += 1

    cap.release()
    writer_vid.release()
    cv2.destroyAllWindows()
    csvf.close()

    # print your occlusion summary
    rate = occluded_frames/total_frames*100 if total_frames>0 else 0
    print(f"→ {occluded_frames}/{total_frames} frames occluded ({rate:.1f}%)")
    print(f" Saved → {out_csv}, {out_vid}")


if __name__=="__main__":
    skipping = args.start is not None
    for pid in sorted(os.listdir(RECS_DIR)):
        recdir   = os.path.join(RECS_DIR, pid)
        if not os.path.isdir(recdir): continue
        print(f"[loop] pid={pid!r}, skipping={skipping}")
        if skipping:
            if pid != args.start:
                continue
            else:
                skipping = False
                print(f"Hit start point: {pid!r}, beginning processing")

        calib_vid = os.path.join(recdir, 'calibration.mp4')
        sess_vid  = os.path.join(recdir, 'session.mp4')
        calib_j   = os.path.join(CAL_DIR, pid + '.json')
        out_csv   = os.path.join(RESULTS_DIR, pid + '_angles.csv')
        out_vid   = os.path.join(RESULTS_DIR, pid + '_annotated.mp4')

        calib = calibrate_from_video(calib_vid, calib_j)
        if calib is None:
            print(f"Calibration skipped for {pid}")
            continue
        track_with_calibration(sess_vid, calib, out_csv, out_vid)
    imu_csv = 'imu_data.csv'   # path to Unity‐recorded IMU output
    fps     = 30.0             # match camera framerate

    for cam_csv in glob.glob(os.path.join(RESULTS_DIR, '*_angles.csv')):
        subj      = os.path.basename(cam_csv).split('_')[0]
        fused_csv = os.path.join(RESULTS_DIR, f'{subj}_fused.csv')

        # call your separate fusion script
        subprocess.run([
            'python', 'fuse_data.py',
            '--camera_csv', cam_csv,
            '--imu_csv',   imu_csv,
            '--output_csv',fused_csv,
            '--fps',       str(fps)
        ], check=True)

    print("All subjects processed and fused.")