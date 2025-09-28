import os

DEFAULT_INDIR = os.path.dirname(__file__)


def discover_sessions_from_indir(indir):
    session_paths = []
    for root, dirs, files in os.walk(indir):
        for file in files:
            if file == 'session.mp4':
                session_paths.append(os.path.join(root, file))
    return session_paths
