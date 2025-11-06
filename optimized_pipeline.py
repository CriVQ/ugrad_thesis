"""
full_pipeline.py

Phase 1: process all session.mp4 under sub*/ -> per-video CSVs, annotated videos, combined Excel
Phase 2: manual baseline picker for every processed video (skips existing baselines)
         applies baseline to per-video CSV immediately
Phase 3: manual segment annotation for every processed video (skips existing segments)
         saves segments CSV per subject

Run:
  python optimized_pipeline.py --indir /path/to/recordings --outdir /path/to/out --model /path/to/metrabs --per-video-csv
"""
import os, argparse, csv, json, math
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.spatial.transform import Rotation as R

# Defaults 
DEFAULT_INDIR       = "recordings"
DEFAULT_OUTDIR      = "test_results2"
DEFAULT_OUTFILE     = "combined_angles.xlsx"
MODEL_PATH          = r'C:/Users/ADAM/Downloads/metrabs/'
SCAPULAR_OFFSET_DEG = 0.0
ELBOW_SUBTRACT_DEG  = 0.0
SHFLEX_SUB_DEG      = 0.0
SHABD_SUB_DEG       = 0.0
CLAMP_DEFAULT       = True
VERBOSE             = False
EPS                 = 1e-9
PROJ_EPS            = 1e-6

#  Geometry & sampling helpers 
def safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.copy() if n < EPS else v / n

def calculate_angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS: return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(na*nb), -1.0, 1.0))))

def signed_angle_from_two_components(forward: float, up: float) -> float:
    return float(np.degrees(np.arctan2(forward, up)))

def signed_angle_from_projection(vec: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> float:
    a = float(np.dot(vec, axis_a)); b = float(np.dot(vec, axis_b))
    return signed_angle_from_two_components(a, b)

def project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    nr = np.linalg.norm(normal)
    if nr < EPS: return v.copy()
    nunit = normal / nr
    return v - np.dot(v, nunit) * nunit

def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    nr = np.linalg.norm(axis)
    if nr < EPS: return v.copy()
    a = axis / nr
    th = math.radians(angle_deg)
    v_par = np.dot(v, a) * a
    v_perp = v - v_par
    return v_par + v_perp * math.cos(th) + np.cross(a, v_perp) * math.sin(th)

def shoulder_origin_axes(p3: np.ndarray, idx: Dict[str,int], scapular_offset_deg: float=0.0, use_fixed_lateral: bool=True):
    up = np.array([0.0,1.0,0.0], dtype=float)
    if use_fixed_lateral:
        lat = np.array([1.0,0.0,0.0], dtype=float)
    else:
        rsho = p3[idx['rsho_smpl']]; lsho = p3[idx['lsho_smpl']]
        lat = project_onto_plane(lsho - rsho, up); lat = safe_normalize(lat)
        if np.linalg.norm(lat) < EPS: lat = np.array([1.0,0.0,0.0], dtype=float)
    fwd = safe_normalize(np.cross(up, lat))
    lat = safe_normalize(np.cross(fwd, up))
    if abs(scapular_offset_deg) > 1e-6:
        lat = rotate_about_axis(lat, up, scapular_offset_deg); fwd = rotate_about_axis(fwd, up, scapular_offset_deg)
        lat, fwd = safe_normalize(lat), safe_normalize(fwd)
    return lat, up, fwd

def sample_from_shoulder(p3: np.ndarray, idx: Dict[str,int],
                         scapular_offset_deg: float=0.0,
                         elbow_subtract_deg: float=ELBOW_SUBTRACT_DEG,
                         clamp_elbow: bool=CLAMP_DEFAULT,
                         shflex_sub_deg: float=SHFLEX_SUB_DEG,
                         shabd_sub_deg: float=SHABD_SUB_DEG,
                         clamp_sh: bool=CLAMP_DEFAULT,
                         use_fixed_lateral: bool=True):
    rsho = p3[idx['rsho_smpl']]; el = p3[idx['relb_smpl']]; wr = p3[idx['rwri_smpl']]
    lat, up, fwd = shoulder_origin_axes(p3, idx, scapular_offset_deg, use_fixed_lateral)
    lat_u, up_u, fwd_u = safe_normalize(lat), safe_normalize(up), safe_normalize(fwd)
    hum_n = safe_normalize(el - rsho)

    # flexion
    proj_sag = project_onto_plane(hum_n, lat_u)
    raw_sf = signed_angle_from_projection(safe_normalize(proj_sag), fwd_u, up_u) if np.linalg.norm(proj_sag) >= PROJ_EPS else 0.0
    corr_sf = (180.0 - raw_sf + 180.0) % 360.0 - 180.0

    # abduction
    proj_fr = project_onto_plane(hum_n, fwd_u)
    raw_sa = signed_angle_from_projection(safe_normalize(proj_fr), lat_u, up_u) if np.linalg.norm(proj_fr) >= PROJ_EPS else 0.0
    corr_sa = (180.0 - raw_sa + 180.0) % 360.0 - 180.0

    adj_shFlex = (corr_sf - shflex_sub_deg + 180.0) % 360.0 - 180.0
    adj_shAbd  = (corr_sa - shabd_sub_deg + 180.0) % 360.0 - 180.0

    fore_n = safe_normalize(wr - el)
    hinge_axis = np.cross(hum_n, fwd_u)
    if np.linalg.norm(hinge_axis) < EPS:
        hinge_axis = np.cross(hum_n, lat_u)
    ha_u = hinge_axis / (np.linalg.norm(hinge_axis) + EPS)
    proj_el = project_onto_plane(fore_n, ha_u)
    raw_elbow = calculate_angle(hum_n, proj_el/np.linalg.norm(proj_el)) if np.linalg.norm(proj_el) >= EPS else calculate_angle(hum_n, fore_n)
    adj_elbow = raw_elbow - elbow_subtract_deg
    if clamp_elbow:
        adj_elbow = float(np.clip(adj_elbow, 0.0, 180.0))

    return raw_elbow, adj_elbow, corr_sf, adj_shFlex, corr_sa, adj_shAbd, (lat_u, up_u, fwd_u)

# I/O helpers
def discover_sessions_from_indir(indir: str) -> List[Tuple[str,str]]:
    found = []
    if not os.path.isdir(indir):
        return found
    for entry in sorted(os.listdir(indir)):
        sub = os.path.join(indir, entry)
        if not os.path.isdir(sub): continue
        candidate = os.path.join(sub, 'session.mp4')
        if os.path.isfile(candidate):
            found.append((os.path.abspath(candidate), f"{entry}_session"))
    return found

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def save_baseline_json(outdir: str, subj: str, baseline: Dict[str,float], frames: List[int], rows: List[Dict]):
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, f"{subj}_baseline.json")
    with open(jpath, 'w') as f: json.dump({'baseline':baseline,'selected_frames':frames}, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{subj}_baseline_frames.csv"), index=False)
    print("Saved baseline:", jpath)
    return jpath

def load_baseline_json(outdir: str, subj: str) -> Optional[Dict]:
    jpath = os.path.join(outdir, f"{subj}_baseline.json")
    if not os.path.isfile(jpath): return None
    try:
        with open(jpath,'r') as f: return json.load(f)
    except: return None

def save_segments_csv(outdir: str, subj: str, segments: List[Tuple[int,int]]):
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, f"{subj}_segments.csv")
    with open(p,'w', newline='') as f:
        w = csv.writer(f); w.writerow(['start_frame','end_frame']); 
        for s,e in segments: w.writerow([s,e])
    print("Saved segments:", p)
    return p

# Model 
def run_model_once(model, img_bgr):
    """Try a few common call patterns and return a prediction object or None."""
    try:
        if hasattr(model, 'detect_poses'):
            try: return model.detect_poses(img_bgr, skeleton='smpl+head_30')
            except Exception: pass
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            try:
                t = tf.convert_to_tensor(img_bgr[np.newaxis,...], dtype=tf.uint8)
                return model.signatures['serving_default'](t)
            except Exception: pass
        try:
            t = tf.convert_to_tensor(img_bgr[np.newaxis,...], dtype=tf.uint8)
            return model(t)
        except Exception: pass
    except Exception:
        return None
    return None

#  Process videos sequentially
def process_video(sess_vid: str, model, idx: Dict[str,int], out_csv: Optional[str], write_annotated: bool,
                  scapular_offset_deg: float, elbow_subtract_deg: float, clamp_elbow: bool,
                  shflex_sub_deg: float, shabd_sub_deg: float, clamp_sh: bool, alpha: float,
                  verbose: bool, session_name_override: Optional[str] = None, use_fixed_lateral: bool = True):
    cap = cv2.VideoCapture(sess_vid)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {sess_vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    writer_vid = None
    if write_annotated:
        out_vid = os.path.splitext(out_csv or sess_vid)[0] + "_annotated.mp4"
        writer_vid = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        if not writer_vid: writer_vid = None

    csvf = None; csv_writer = None
    if out_csv:
        try:
            csvf = open(out_csv,'w',newline=''); csv_writer = csv.writer(csvf)
            csv_writer.writerow(['frame','time_s','raw_elbow_deg','adj_elbow_deg','raw_shFlex_deg','adj_shFlex_deg','raw_shAbd_deg','adj_shAbd_deg','q_sho_x','q_sho_y','q_sho_z','q_sho_w','occluded'])
        except Exception as e:
            print("CSV open failed:", e); csvf = None; csv_writer = None

    elbow_rows, shf_rows, sha_rows = [], [], []
    frame_idx = 0
    rsho_prev = None

    # show window optionally for annotation check 
    can_show = False
    try:
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL); can_show = True
    except: pass

    while True:
        ok, frame = cap.read()
        if not ok: break
        time_s = frame_idx / fps

        pred = run_model_once(model, frame)
        p3_arr = None; p2_arr = None
        if pred is None:
            p3_arr = None
        else:
            try:
                if isinstance(pred, dict):
                    for k in ('poses3d','poses_3d','smpl_joints','smpl_world','smpl_xyz'):
                        if k in pred:
                            arr = pred[k]; p3_arr = arr.numpy() if hasattr(arr,'numpy') else np.array(arr); break
                    for k in ('poses2d','poses_2d','keypoints2d','keypoints'):
                        if k in pred:
                            arr2 = pred[k]; p2_arr = arr2.numpy() if hasattr(arr2,'numpy') else np.array(arr2); break
                else:
                    arr = pred.numpy() if hasattr(pred,'numpy') else np.array(pred)
                    if arr.ndim >= 3 and arr.shape[-1] == 3: p3_arr = arr
            except Exception:
                p3_arr = None

        if p3_arr is None:
            # occlusion row
            elbow_rows.append({'frame':frame_idx,'time_s':time_s,'raw_elbow_deg':np.nan,'adj_elbow_deg':np.nan,'session':session_name_override or os.path.splitext(os.path.basename(sess_vid))[0]})
            shf_rows.append({'frame':frame_idx,'time_s':time_s,'raw_shFlex_deg':np.nan,'adj_shFlex_deg':np.nan,'session':session_name_override or os.path.splitext(os.path.basename(sess_vid))[0]})
            sha_rows.append({'frame':frame_idx,'time_s':time_s,'raw_shAbd_deg':np.nan,'adj_shAbd_deg':np.nan,'session':session_name_override or os.path.splitext(os.path.basename(sess_vid))[0]})
            if csv_writer: csv_writer.writerow([frame_idx, f"{time_s:.3f}"] + ['']*9 + [1])
            if writer_vid: writer_vid.write(frame)
            if can_show: 
                cv2.imshow('Tracking', frame); 
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            frame_idx += 1; continue

        # pick first detection
        p3_frame = p3_arr[0] if (p3_arr.ndim == 3 and p3_arr.shape[0] >= 1) else p3_arr
        p2_frame = p2_arr[0] if (p2_arr is not None and p2_arr.ndim == 3 and p2_arr.shape[0] >= 1) else p2_arr if p2_arr is not None else None

        p3 = p3_frame.astype(float)
        try:
            if p3.shape[1] >= max(idx.values())+1:
                p3[...,1], p3[...,2] = p3[...,2].copy(), -p3[...,1].copy()
        except: pass

        try:
            rsho_curr = p3[idx['rsho_smpl']].astype(float)
        except:
            rsho_curr = np.zeros(3,dtype=float)
        if rsho_prev is None: rsho_smooth = rsho_curr
        else: rsho_smooth = 0.8 * rsho_prev + (1.0 - 0.8) * rsho_curr
        rsho_prev = rsho_smooth

        p3_local = p3 - rsho_smooth

        try:
            raw_elbow, adj_elbow, raw_sf, adj_sf, raw_sa, adj_sa, axes = sample_from_shoulder(p3_local, idx, scapular_offset_deg, elbow_subtract_deg, clamp_elbow, shflex_sub_deg, shabd_sub_deg, clamp_sh, use_fixed_lateral)
        except Exception:
            raw_elbow = adj_elbow = raw_sf = adj_sf = raw_sa = adj_sa = np.nan; axes=(np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]))

        lat, up, fwd = axes
        Rm = np.column_stack((lat, up, fwd))
        try:
            if not np.isfinite(np.linalg.det(Rm)) or abs(np.linalg.det(Rm)-1.0) > 1e-2: raise Exception
            q = R.from_matrix(Rm).as_quat()
        except:
            q = np.array([0.0,0.0,0.0,1.0])

        session_col = session_name_override or os.path.splitext(os.path.basename(sess_vid))[0]
        elbow_rows.append({'frame':frame_idx,'time_s':time_s,'raw_elbow_deg':raw_elbow,'adj_elbow_deg':adj_elbow,'session':session_col})
        shf_rows.append({'frame':frame_idx,'time_s':time_s,'raw_shFlex_deg':raw_sf,'adj_shFlex_deg':adj_sf,'session':session_col})
        sha_rows.append({'frame':frame_idx,'time_s':time_s,'raw_shAbd_deg':raw_sa,'adj_shAbd_deg':adj_sa,'session':session_col})

        if csv_writer:
            csv_writer.writerow([frame_idx, f"{time_s:.3f}", f"{raw_elbow:.6f}", f"{adj_elbow:.6f}", f"{raw_sf:.6f}", f"{adj_sf:.6f}", f"{raw_sa:.6f}", f"{adj_sa:.6f}", f"{q[0]:.6f}", f"{q[1]:.6f}", f"{q[2]:.6f}", f"{q[3]:.6f}", 0])

        if p2_frame is not None:
            try:
                for a,b in ((idx['rsho_smpl'], idx['relb_smpl']), (idx['relb_smpl'], idx['rwri_smpl'])):
                    pa = tuple(int(x) for x in p2_frame[a]); pb = tuple(int(x) for x in p2_frame[b]); cv2.line(frame, pa, pb, (0,255,0), 2)
                for j in ('rsho_smpl','relb_smpl','rwri_smpl'):
                    pos = tuple(int(x) for x in p2_frame[idx[j]]); cv2.circle(frame, pos, 4, (0,0,255), -1)
            except: pass

        cv2.putText(frame, f"E(raw/adj):{raw_elbow:.1f}/{adj_elbow:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"SF(raw/adj):{raw_sf:.1f}/{adj_sf:.1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"SA(raw/adj):{raw_sa:.1f}/{adj_sa:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        if writer_vid: writer_vid.write(frame)
        if can_show:
            cv2.imshow('Tracking', frame); 
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_idx += 1

    cap.release()
    if writer_vid: writer_vid.release()
    if csvf: csvf.close()
    if can_show: cv2.destroyAllWindows()

    df_el = pd.DataFrame(elbow_rows) if elbow_rows else pd.DataFrame(columns=['frame','time_s','raw_elbow_deg','adj_elbow_deg','session'])
    df_shf = pd.DataFrame(shf_rows) if shf_rows else pd.DataFrame(columns=['frame','time_s','raw_shFlex_deg','adj_shFlex_deg','session'])
    df_sha = pd.DataFrame(sha_rows) if sha_rows else pd.DataFrame(columns=['frame','time_s','raw_shAbd_deg','adj_shAbd_deg','session'])
    return df_el, df_shf, df_sha

# Apply baseline adjustments 
def apply_baseline_to_csv(per_csv: str, baseline: Dict[str,float]):
    """Loads per_csv, subtracts baseline from adjusted columns, writes back (backup original)."""
    if not os.path.isfile(per_csv):
        print("Per-video CSV not found for baseline application:", per_csv); return False
    df = safe_read_csv(per_csv)
    if df is None:
        print("Failed to read per-video CSV:", per_csv); return False
    # backup original
    bak = per_csv + ".orig"
    if not os.path.exists(bak):
        try: os.rename(per_csv, bak); df.to_csv(per_csv, index=False)  # we still want the file path; save below after adjustments
        except: pass
    # columns expected: adj_elbow_deg, adj_shFlex_deg, adj_shAbd_deg
    for col, key in (('adj_elbow_deg','adj_elbow_deg'), ('adj_shFlex_deg','adj_shFlex_deg'), ('adj_shAbd_deg','adj_shAbd_deg')):
        if col in df.columns and key in baseline:
            # subtract baseline value
            df[col] = df[col].astype(float) - float(baseline[key])
    df.to_csv(per_csv, index=False)
    print("Applied baseline to per-video CSV:", per_csv)
    return True

def apply_baselines_to_combined_xl(combined_xl: str, csv_dir: str):
    """Reads combined Excel and applies any <sub>_baseline.json found in csv_dir (session names used to match)."""
    if not os.path.isfile(combined_xl): 
        print("Combined Excel not found:", combined_xl); return False
    sheets = pd.read_excel(combined_xl, sheet_name=None)
    # load all baseline jsons in csv_dir
    baselines = {}
    for fname in os.listdir(csv_dir):
        if fname.endswith('_baseline.json'):
            try:
                data = json.load(open(os.path.join(csv_dir,fname),'r'))
                subj = fname.replace('_baseline.json','')
                baselines[subj] = data.get('baseline',{})
            except Exception:
                pass
    if not baselines:
        print("No baseline JSONs found to apply to combined workbook.")
        return False

    # apply per-sheet by matching 'session' or folder prefix
    for sheet_name, df in sheets.items():
        if 'session' not in df.columns:
            continue
        df2 = df.copy()
        def apply_row(row):
            sess = str(row.get('session',''))
            # session in per-video CSVs used format like 'sub01_session'; baseline files use 'sub01' as subject
            subj = sess.split('_')[0] if '_' in sess else sess
            baseline = baselines.get(subj)
            if baseline:
                # apply to adj columns if present
                if 'adj_elbow_deg' in df2.columns and 'adj_elbow_deg' in baseline:
                    row['adj_elbow_deg'] = float(row.get('adj_elbow_deg',0.0)) - float(baseline['adj_elbow_deg'])
                if 'signed_shFlex_deg' in df2.columns and 'adj_shFlex_deg' in baseline:
                    row['signed_shFlex_deg'] = float(row.get('signed_shFlex_deg',0.0)) - float(baseline['adj_shFlex_deg'])
                if 'signed_shAbd_deg' in df2.columns and 'adj_shAbd_deg' in baseline:
                    row['signed_shAbd_deg'] = float(row.get('signed_shAbd_deg',0.0)) - float(baseline['adj_shAbd_deg'])
            return row
        sheets[sheet_name] = df2.apply(apply_row, axis=1)

    # write back
    with pd.ExcelWriter(combined_xl) as writer:
        for sn, df2 in sheets.items():
            df2.to_excel(writer, sheet_name=sn, index=False)
    print("Applied baselines to combined workbook:", combined_xl)
    return True

# Interactive baseline picker & segment annotator 
def run_baseline_picker(video_path: str, per_video_csv: Optional[str], out_dir: str):
    """Interactive picker — identical behavior to earlier versions but with resume semantics."""
    print(f"\n=== BASELINE PICKER ===\nVideo: {video_path}\n")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video:", video_path); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frame_map = {}
    if per_video_csv and os.path.exists(per_video_csv):
        df = safe_read_csv(per_video_csv)
        if df is not None and 'frame' in df.columns:
            frame_map = {int(r['frame']): r for _, r in df.iterrows()}

    subj = os.path.basename(os.path.dirname(video_path))
    baseline_existing = load_baseline_json(out_dir, subj)
    if baseline_existing:
        print(f"Existing baseline found for {subj}. Options: [k]eep / [r]edo / [s]kip this subject.")
        while True:
            c = input("k/r/s: ").strip().lower()
            if c == 'k':
                print("Keeping existing baseline and applying it to per-video CSV.")
                apply_baseline_to_csv(per_video_csv, baseline_existing['baseline'])
                return baseline_existing['baseline']
            if c == 's':
                print("Skipping baseline for", subj); return None
            if c == 'r':
                print("Redoing baseline for", subj); break

    idx = 0; selected = set()
    window = "Baseline picker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    def show(i):
        i = max(0, min(i, total_frames-1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: frame = np.ones((480,640,3), dtype=np.uint8)*255
        info = f"Frame {i}/{total_frames-1}  Selected:{len(selected)}"
        r = frame_map.get(i)
        if r is not None:
            info += f"  SF:{r.get('adj_shFlex_deg', r.get('signed_shFlex_deg'))} SA:{r.get('adj_shAbd_deg', r.get('signed_shAbd_deg'))} E:{r.get('adj_elbow_deg', r.get('signed_elbow_deg'))}"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if i in selected: cv2.rectangle(frame, (5,5), (200,55), (0,128,0), 2)
        cv2.imshow(window, frame)

    show(idx)
    baseline = None
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Quit without saving baseline."); baseline = None; break
        elif key in (ord('d'), 83): idx += 1
        elif key in (ord('a'), 81): idx -= 1
        elif key == ord('1'): idx += 1
        elif key == ord('2'): idx += 5
        elif key == ord('3'): idx += 10
        elif key == ord('4'): idx += 15
        elif key == ord('5'): idx -= 5
        elif key == ord('6'): idx -= 10
        elif key == ord('7'): idx -= 15
        elif key == ord('c'):
            if idx in selected: selected.remove(idx); print("Removed", idx)
            else: selected.add(idx); print("Captured", idx)
        elif key == ord('p'): print("Selected frames:", sorted(selected))
        elif key == ord('s'):
            sel_sorted = sorted(selected)
            if not sel_sorted:
                print("No frames selected. Choose frames first."); continue
            rows, present = [], []
            for fr in sel_sorted:
                r = frame_map.get(fr)
                if r is None: continue
                rows.append({'frame':fr, 'signed_shFlex_deg': r.get('adj_shFlex_deg', r.get('signed_shFlex_deg')), 'signed_shAbd_deg': r.get('adj_shAbd_deg', r.get('signed_shAbd_deg')), 'signed_elbow_deg': r.get('adj_elbow_deg', r.get('signed_elbow_deg'))})
                present.append(fr)
            if not present:
                confirm = input("No angles found for selected frames. Save zero baseline? (y/n): ").strip().lower()
                if confirm != 'y': continue
                baseline = {'adj_elbow_deg':0.0,'adj_shFlex_deg':0.0,'adj_shAbd_deg':0.0}
            else:
                import numpy as _np
                el = _np.array([r['signed_elbow_deg'] for r in rows if r['signed_elbow_deg'] is not None], dtype=float)
                sf = _np.array([r['signed_shFlex_deg'] for r in rows if r['signed_shFlex_deg'] is not None], dtype=float)
                sa = _np.array([r['signed_shAbd_deg'] for r in rows if r['signed_shAbd_deg'] is not None], dtype=float)
                baseline = {'adj_elbow_deg': float(_np.nanmedian(el)) if el.size else 0.0,
                            'adj_shFlex_deg': float(_np.nanmedian(sf)) if sf.size else 0.0,
                            'adj_shAbd_deg': float(_np.nanmedian(sa)) if sa.size else 0.0}
            # save baseline and apply to per-video CSV
            subj_name = os.path.basename(os.path.dirname(video_path))
            save_baseline_json(out_dir, subj_name, baseline, sel_sorted, rows)
            if per_video_csv:
                apply_baseline_to_csv(per_video_csv, baseline)
            break
        else:
            print("Unhandled key", key)
        show(idx)

    cap.release(); cv2.destroyWindow(window)
    return baseline

def run_segment_annotator(video_path: str, out_dir: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("Cannot open", video_path); return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    subj = os.path.basename(os.path.dirname(video_path))
    seg_path = os.path.join(out_dir, f"{subj}_segments.csv")
    if os.path.isfile(seg_path):
        print(f"Segments file exists for {subj}. [k]eep / [r]edo / [s]kip?")
        while True:
            c = input("k/r/s: ").strip().lower()
            if c == 'k': print("Keeping existing segments."); return None
            if c == 's': print("Skipping."); return None
            if c == 'r': break

    idx = 0; segments = []; cur_start = None
    window = "Segment annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    def show(i):
        i = max(0, min(i, total_frames-1)); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, frame = cap.read()
        if not ok: frame = np.ones((480,640,3), dtype=np.uint8)*255
        info = f"Frame {i}/{total_frames-1} Segments:{len(segments)}"
        if cur_start is not None: info += f" (marking start {cur_start})"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow(window, frame)
    show(idx)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'): print("Quit segment annotator without saving."); break
        elif key == ord('d'): idx += 1
        elif key == ord('a'): idx -= 1
        elif key == ord('b'): cur_start = idx; print("Marked start", idx)
        elif key == ord('e'):
            if cur_start is None: print("Mark start first (b)")
            else: segments.append((int(cur_start), int(idx))); print("Added segment", cur_start, idx); cur_start = None
        elif key == ord('p'): print("Segments:", segments)
        elif key == ord('s'): save_segments_csv(out_dir, subj, segments); break
        else: print("Unhandled key", key)
        show(idx)
    cap.release(); cv2.destroyWindow(window)
    return segments

# Main 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=DEFAULT_INDIR)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--per-video-csv", action='store_true')
    ap.add_argument("--verbose", action='store_true')
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    sessions = discover_sessions_from_indir(args.indir)
    if not sessions:
        raise SystemExit("No session.mp4 files found under --indir (expected path/subXX/session.mp4)")

    os.makedirs(args.outdir, exist_ok=True)
    csv_dir = os.path.join(args.outdir, "per_video_csvs"); os.makedirs(csv_dir, exist_ok=True)
    combined_xl = os.path.join(args.outdir, DEFAULT_OUTFILE)

    print("Loading model from", args.model)
    model = hub.load(args.model)
    raw_names = model.per_skeleton_joint_names['smpl+head_30'].numpy()
    jn = [x.decode('utf-8') if isinstance(x,(bytes,bytearray)) else str(x) for x in raw_names]
    idx = { 'rsho_smpl': jn.index('rsho_smpl'), 'lsho_smpl': jn.index('lsho_smpl'), 'relb_smpl': jn.index('relb_smpl'), 'rwri_smpl': jn.index('rwri_smpl'), 'neck_smpl': jn.index('neck_smpl') }

    # PHASE 1: process ALL videos 
    print("\\n=== PHASE 1: Processing videos ===")
    el_list, sf_list, sa_list = [], [], []
    for vid_path, session_name in sessions:
        print("Processing:", vid_path)
        per_csv = os.path.join(csv_dir, f"{session_name}_angles.csv") if args.per_video_csv else os.path.join(csv_dir, f"{session_name}_angles.csv")
        df_el, df_sf, df_sa = process_video(sess_vid=vid_path, model=model, idx=idx, out_csv=per_csv, write_annotated=True,
                                            scapular_offset_deg=SCAPULAR_OFFSET_DEG, elbow_subtract_deg=ELBOW_SUBTRACT_DEG,
                                            clamp_elbow=CLAMP_DEFAULT, shflex_sub_deg=SHFLEX_SUB_DEG, shabd_sub_deg=SHABD_SUB_DEG,
                                            clamp_sh=CLAMP_DEFAULT, alpha=0.8, verbose=VERBOSE, session_name_override=session_name, use_fixed_lateral=True)
        el_list.append(df_el.assign(session=session_name))
        sf_list.append(df_sf.assign(session=session_name))
        sa_list.append(df_sa.assign(session=session_name))

    # write combined Excel (unadjusted initially)
    with pd.ExcelWriter(combined_xl) as writer:
        pd.concat(el_list, ignore_index=True).to_excel(writer, sheet_name="elbow", index=False)
        pd.concat(sf_list, ignore_index=True).to_excel(writer, sheet_name="shflex", index=False)
        pd.concat(sa_list, ignore_index=True).to_excel(writer, sheet_name="shabd", index=False)
    print("Combined workbook written:", combined_xl)

    #  PHASE 2: baseline picker for ALL videos (can resume) 
    print("\\n=== PHASE 2: Baseline picking (for all videos) ===")
    for vid_path, session_name in sessions:
        per_csv = os.path.join(csv_dir, f"{session_name}_angles.csv")
        print(f"\\n-> Baseline picker for {session_name}")
        baseline = run_baseline_picker(video_path=vid_path, per_video_csv=per_csv, out_dir=csv_dir)
        if baseline is not None:
            print("Baseline saved & applied for", session_name)
        else:
            print("Baseline skipped or kept existing for", session_name)

    # After all baselines saved, apply baselines to combined workbook
    print("\\nApplying any saved baselines to combined workbook...")
    apply_baselines_to_combined_xl(combined_xl, csv_dir)

    # PHASE 3: segment annotator for ALL videos (can resume) 
    print("\\n=== PHASE 3: Segment annotation (for all videos) ===")
    for vid_path, session_name in sessions:
        print(f"\\n-> Segment annotator for {session_name}")
        run_segment_annotator(video_path=vid_path, out_dir=csv_dir)

    print("\\nAll phases complete. Outputs are in:", args.outdir)
    print("Per-video CSVs & baseline/segments JSON/CSVs are in:", csv_dir)

if __name__ == '__main__':
    main()
