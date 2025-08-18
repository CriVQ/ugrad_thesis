import os, cv2, json, csv, numpy as np, tensorflow as tf, tensorflow_hub as hub

# ─────────────────────────────────────────────────────
# 1) CONFIG & MODEL
# ─────────────────────────────────────────────────────
RECS_DIR   = 'recordings'
CAL_DIR    = 'calibrations'
RESULTS_DIR= 'results'
TRUE_MAX   = 150.0
MODEL_PATH = 'C:/Users/ADAM/Downloads/metrabs/'

model      = hub.load(MODEL_PATH)
jn         = model.per_skeleton_joint_names['smpl+head_30'].numpy().astype(str).tolist()
idx        = { name: jn.index(name) for name in
              ('rsho_smpl','relb_smpl','rwri_smpl',
               'lsho_smpl','rhip_smpl','lhip_smpl') }

# ─────────────────────────────────────────────────────
# 2) MATH HELPERS
# ─────────────────────────────────────────────────────
def calculate_angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return 0.0
    c = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def midpoint(a, b):
    return (a + b) * 0.5

# ─────────────────────────────────────────────────────
# 3) CALIBRATE FROM VIDEO (5 POSES)
# ─────────────────────────────────────────────────────
PROMPTS = [
    ("Elbow 0° (straight)",    'elbow0'),
    ("Elbow max flex",         'elbowMax'),
    ("Shoulder 0° (down)",     'sh0'),
    ("Shoulder max flex (front)", 'shFlexMax'),
    ("Shoulder max abd (side)",   'shAbdMax'),
]

def sample_raws(p3):
    # compute raw elbow, raw shoulder-flex, raw shoulder-abd
    sh = p3[idx['rsho_smpl']]; el = p3[idx['relb_smpl']]; wr = p3[idx['rwri_smpl']]
    upper, lower = el - sh, wr - el
    raw_e = calculate_angle(upper, lower)

    sho_mid = midpoint(p3[idx['rsho_smpl']], p3[idx['lsho_smpl']])
    hip_mid = midpoint(p3[idx['rhip_smpl']], p3[idx['lhip_smpl']])
    up      = sho_mid - hip_mid; up /= np.linalg.norm(up)+1e-9

    lat     = p3[idx['lsho_smpl']] - p3[idx['rsho_smpl']]; lat /= np.linalg.norm(lat)+1e-9
    sag     = upper - np.dot(upper, lat)*lat
    raw_sf  = calculate_angle(sag, up)

    fwd     = np.cross(up, lat); fwd /= np.linalg.norm(fwd)+1e-9
    cor     = upper - np.dot(upper, fwd)*fwd
    raw_sa  = calculate_angle(cor, up)

    return raw_e, raw_sf, raw_sa

def calibrate_from_video(calib_vid, calib_json):
    if os.path.exists(calib_json):
        with open(calib_json) as f:
            print(f"Loaded {calib_json}")
            return json.load(f)

    cap = cv2.VideoCapture(calib_vid)
    cv2.namedWindow('Calibrate', cv2.WINDOW_NORMAL)
    data = {k: [] for _,k in PROMPTS}

    for text, key in PROMPTS:
        print(f"\n=== Hold: {text} and press C ===")
        # wait for your key press
        while True:
            ok, frame = cap.read()
            if not ok: break
            cv2.putText(frame, f"{text} (C)", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Calibrate', frame)
            if cv2.waitKey(1)&0xFF == ord('c'):
                break
        # sample next 60 frames
        for _ in range(60):
            ok, f2 = cap.read()
            if not ok: continue
            tfimg = tf.image.convert_image_dtype(f2, tf.uint8)
            pred  = model.detect_poses(tfimg, skeleton='smpl+head_30')
            p3s   = pred['poses3d'].numpy()
            if p3s.shape[0]==0: continue
            p3 = p3s[0]; p3[...,1], p3[...,2] = p3[...,2], -p3[...,1]
            raw_e, raw_sf, raw_sa = sample_raws(p3)
            if key.startswith('elbow'):
                data[key].append(raw_e)
            elif key == 'sh0':
                data[key].append(raw_sf)
            elif key == 'shFlexMax':
                data[key].append(raw_sf)
            elif key == 'shAbdMax':
                data[key].append(raw_sa)

    cap.release()
    cv2.destroyAllWindows()

    # compute base & scale for each DOF
    b_e  = float(min(data['elbow0']));   m_e  = float(max(data['elbowMax']))
    b_s0 = float(min(data['sh0']));      m_sf = float(max(data['shFlexMax']))
    m_sa = float(max(data['shAbdMax']))

    calib = {
      'base_elbow': b_e,  'scale_elbow': TRUE_MAX/(m_e - b_e),
      'base_sh0':   b_s0, 'scale_shflex': TRUE_MAX/(m_sf - b_s0),
                         'scale_shabd':  TRUE_MAX/(m_sa - b_s0)
    }

    os.makedirs(os.path.dirname(calib_json), exist_ok=True)
    with open(calib_json, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration → {calib_json}")
    return calib

# ─────────────────────────────────────────────────────
# 4) TRACK WITH CALIBRATION
# ─────────────────────────────────────────────────────
def track_with_calibration(sess_vid, calib, out_csv, out_vid):
    cap = cv2.VideoCapture(sess_vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_vid = cv2.VideoWriter(out_vid, fourcc, fps, (w,h))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    csvf = open(out_csv, 'w', newline='')
    writer = csv.writer(csvf)
    writer.writerow(['frame',
                     'raw_elbow','true_elbow',
                     'raw_shFlex','true_shFlex',
                     'raw_shAbd','true_shAbd'])
    frame = 0
    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
    print(f"Processing session → {sess_vid}")

    while True:
        ok, img = cap.read()
        if not ok: break

        tfimg = tf.image.convert_image_dtype(img, tf.uint8)
        pred  = model.detect_poses(tfimg, skeleton='smpl+head_30')
        p3s   = pred['poses3d'].numpy()
        p2s   = pred['poses2d'].numpy() if pred['poses2d'] is not None else None

        if p3s.shape[0]>0 and p2s is not None:
            p3    = p3s[0]; p3[...,1],p3[...,2] = p3[...,2],-p3[...,1]
            raw_e, raw_sf, raw_sa = sample_raws(p3)

            # apply calibration
            el = np.clip((raw_e  - calib['base_elbow'])*calib['scale_elbow'],   0, TRUE_MAX)
            sf = np.clip((raw_sf - calib['base_sh0']  )*calib['scale_shflex'],  0, TRUE_MAX)
            sa = np.clip((raw_sa - calib['base_sh0']  )*calib['scale_shabd'],   0, TRUE_MAX)

            # draw right arm
            p2 = p2s[0]
            for a,b in ((idx['rsho_smpl'],idx['relb_smpl']),
                        (idx['relb_smpl'],idx['rwri_smpl'])):
                cv2.line(img,
                         tuple(p2[a].astype(int)),
                         tuple(p2[b].astype(int)),
                         (0,255,0), 2)
            for j in ('rsho_smpl','relb_smpl','rwri_smpl'):
                cv2.circle(img,
                           tuple(p2[idx[j]].astype(int)),
                           5, (0,0,255), -1)

            # overlay all three
            cv2.putText(img, f"E: {el:.1f}°",  (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(img, f"SF:{sf:.1f}°",  (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(img, f"SA:{sa:.1f}°",  (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            writer.writerow([frame, raw_e, el, raw_sf, sf, raw_sa, sa])

        writer_vid.write(img)
        cv2.imshow('Tracking', img)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        frame += 1

    cap.release(); writer_vid.release()
    cv2.destroyAllWindows(); csvf.close()
    print(f"Saved → {out_csv}, {out_vid}")

# ─────────────────────────────────────────────────────
# 5) BATCH RUN ALL SUBJECTS
# ─────────────────────────────────────────────────────
if __name__=="__main__":
    for pid in sorted(os.listdir(RECS_DIR)):
        recdir   = os.path.join(RECS_DIR, pid)
        if not os.path.isdir(recdir): continue
        
        calib_vid = os.path.join(recdir, 'calib.mp4')
        sess_vid  = os.path.join(recdir, 'session.mp4')
        calib_j   = os.path.join(CAL_DIR, pid + '.json')
        out_csv   = os.path.join(RESULTS_DIR, pid + '_angles.csv')
        out_vid   = os.path.join(RESULTS_DIR, pid + '_annotated.mp4')

        calib = calibrate_from_video(calib_vid, calib_j)
        track_with_calibration(sess_vid, calib, out_csv, out_vid)
