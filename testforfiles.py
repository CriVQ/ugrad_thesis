"""
Standalone test for discover_sessions_from_indir()
"""

import os
import sys

# so we import the local copy of optimized_pipeline.py
sys.path.insert(0, os.path.dirname(__file__))

try:
    from track_shoulder_combined_dir import discover_sessions_from_indir
except ImportError as e:
    print("ERROR: could not import discover_sessions_from_indir:", e)
    sys.exit(1)

# Override the recordings path on the command line
if len(sys.argv) > 1:
    recordings_dir = sys.argv[1]
else:
    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")

print(f"→ Recordings directory resolved to: {recordings_dir!r}")
print("→ Exists?", os.path.exists(recordings_dir))
print("→ Is a dir?", os.path.isdir(recordings_dir))
if os.path.isdir(recordings_dir):
    print("→ Contents:", os.listdir(recordings_dir))

# Call the function
sessions = discover_sessions_from_indir(recordings_dir)
print("→ discover_sessions_from_indir() returned:", sessions)
if not sessions:
    print("→ NO sessions found. Double-check that recordings_dir contains subfolders with mp4 files.")
else:
    print(f"→ Found {len(sessions)} session(s):")
    for vid, name in sessions:
        print(f"   • {name!r} → {vid!r}")