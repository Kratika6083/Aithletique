import numpy as np
import os

def load_motion_reference(path):
    """
    Load saved motion sequence from .npz file.
    """
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data['landmarks']

def compute_motion_similarity(live_seq, ref_seq):
    """
    Compare live motion sequence to reference motion.
    Returns an accuracy score.
    """
    min_len = min(len(live_seq), len(ref_seq))
    if min_len == 0:
        return 0

    live_seq = live_seq[-min_len:]
    ref_seq = ref_seq[:min_len]

    diffs = []
    for i in range(min_len):
        live_frame = live_seq[i]
        ref_frame = ref_seq[i]

        if len(live_frame) != len(ref_frame):
            continue

        diff = np.linalg.norm(np.array(live_frame) - np.array(ref_frame))
        diffs.append(diff)

    if len(diffs) == 0:
        return 0

    avg_diff = np.mean(diffs)
    normalized = max(0, 100 - avg_diff * 2)
    return normalized
