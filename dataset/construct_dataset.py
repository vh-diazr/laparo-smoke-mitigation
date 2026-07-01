"""
construct_dataset.py
====================
Training dataset construction for the hybrid physical-neural model for
nonhomogeneous smoke mitigation in laparoscopic images.

Scientific context
------------------
This script assembles the supervised training dataset from the synthetic
smoke-degraded laparoscopic video triplets produced by the simulation
pipeline (smoke_simulation.py → apply_smoke_degradation.py). Each triplet
consists of:

  - degraded_sequence{N}.mp4     : synthetically smoke-degraded frames f(x,y)
  - transmission_sequence{N}.mp4 : ground-truth transmission maps T(x,y)
  - undegraded_sequence{N}.mp4   : ground-truth smoke-free frames J(x,y)

For each frame, the script resizes all three channels to the target
resolution (480 × 512 pixels, divisible by 16 as required by the
CNN encoder downsampling factor), converts degraded and clean frames
from BGR to RGB for PyTorch compatibility, and extracts the transmission
map from the R channel of the transmission video to avoid chroma averaging
artefacts introduced by lossy video compression.

The resulting arrays are saved as NumPy binary files (.npy) and consumed
directly by the model training script (training/train.py).

Pipeline position
-----------------
  apply_smoke_degradation.py
          ↓
  {degraded_sequence{N}.mp4,
   transmission_sequence{N}.mp4,
   undegraded_sequence{N}.mp4}
          ↓
  construct_dataset.py
          ↓
  {laparo_images_deg_x.npy,
   laparo_images_clean_j.npy,
   trans_funcs_t.npy,
   frames_per_video_used.npy,
   video_ids_used.npy}

Output arrays
-------------
  laparo_images_deg_x.npy   : (N, 480, 512, 3) uint8 — degraded frames
  laparo_images_clean_j.npy : (N, 480, 512, 3) uint8 — clean reference frames
  trans_funcs_t.npy         : (N, 480, 512)    uint8 — transmission maps
  frames_per_video_used.npy : (V,)             int64 — frames per video
  video_ids_used.npy        : (V,)             int64 — video sequence indices

Prerequisites
-------------
  pip install numpy opencv-python

Reference
---------
  Diaz-Ramirez et al., "Nonhomogeneous smoke mitigation in laparoscopic
  images using a hybrid physical-neural model", Medical & Biological
  Engineering & Computing, Springer Nature, 2025. (Under review.)

Usage
-----
  1. Set home_path, data_path, and out_path below to match your directory
     structure.
  2. Run: python construct_dataset.py
"""

import numpy as np
import cv2
import os

# =============================================================================
# Configuration
# =============================================================================

# Set your path here
home_path = '/your/path/to/laparo_dehazing/'
data_path = os.path.join(home_path, 'train_data')   # input triplet videos
out_path  = os.path.join(home_path, 'full_train_data')  # output .npy arrays
os.makedirs(out_path, exist_ok=True)

# Target resolution — must be divisible by 16 (CNN encoder downsampling factor)
TARGET_H = 480
TARGET_W = 512

assert TARGET_H % 16 == 0 and TARGET_W % 16 == 0, \
    f"TARGET_H and TARGET_W must be divisible by 16 (got {TARGET_H}×{TARGET_W})"

# Number of laparoscopic video sequences used for training dataset construction
Nvids = 30

# =============================================================================
# Step 1 — Collect existing video triplets
# =============================================================================

print(f"data_path : {data_path}")
print(f"out_path  : {out_path}")
print(f"Target resolution: {TARGET_H}×{TARGET_W}")

video_paths = []

for ind in range(Nvids):
    seq       = ind + 1
    vid_deg   = os.path.join(data_path, f'degraded_sequence{seq}.mp4')
    vid_t     = os.path.join(data_path, f'transmission_sequence{seq}.mp4')
    vid_clean = os.path.join(data_path, f'undegraded_sequence{seq}.mp4')

    if not (os.path.exists(vid_deg) and
            os.path.exists(vid_t)   and
            os.path.exists(vid_clean)):
        print(f"  Video {seq}: missing file(s), skipped.")
        continue

    video_paths.append((seq, vid_deg, vid_t, vid_clean))

print(f"Found {len(video_paths)} complete video triplets.")

# =============================================================================
# Step 2 — First pass: estimate total frame count and verify source resolution
# =============================================================================

total_frames = 0
H = W = None

for seq, vid_deg, vid_t, vid_clean in video_paths:
    cap        = cv2.VideoCapture(vid_deg)
    n          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  Video {seq}: could not read first frame, skipped.")
        continue

    if H is None:
        H, W = frame.shape[:2]
        print(f"Source resolution detected: {H}×{W}")
    else:
        assert frame.shape[:2] == (H, W), \
            f"Video {seq} resolution mismatch: expected {H}×{W}, " \
            f"got {frame.shape[:2]}"

    total_frames += n
    print(f"  Video {seq}: ~{n} frames")

print(f"\nTotal frames (estimated): {total_frames}")
print(f"Allocating arrays ...")

# =============================================================================
# Step 3 — Preallocate output arrays at target resolution
# =============================================================================

arr_deg   = np.empty((total_frames, TARGET_H, TARGET_W, 3), dtype=np.uint8)
arr_clean = np.empty((total_frames, TARGET_H, TARGET_W, 3), dtype=np.uint8)
arr_t     = np.empty((total_frames, TARGET_H, TARGET_W),    dtype=np.uint8)

# =============================================================================
# Step 4 — Second pass: read, resize, convert, and store frames
# =============================================================================

idx              = 0
frames_per_video = []   # true frame count per video
video_ids_used   = []   # processed video sequence indices

for seq, vid_deg, vid_t, vid_clean in video_paths:
    cap_deg   = cv2.VideoCapture(vid_deg)
    cap_t     = cv2.VideoCapture(vid_t)
    cap_clean = cv2.VideoCapture(vid_clean)

    n_deg   = int(cap_deg.get(cv2.CAP_PROP_FRAME_COUNT))
    n_t     = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    n_clean = int(cap_clean.get(cv2.CAP_PROP_FRAME_COUNT))

    if not (n_deg == n_t == n_clean):
        print(f"  WARNING Video {seq}: frame count mismatch "
              f"deg={n_deg} / t={n_t} / clean={n_clean} — "
              f"stopping at shortest stream.")

    total_ok = 0

    while True:
        ret_deg,   frame_deg   = cap_deg.read()
        ret_t,     frame_t     = cap_t.read()
        ret_clean, frame_clean = cap_clean.read()

        if not (ret_deg and ret_t and ret_clean):
            break

        # Resize to target resolution (INTER_AREA minimizes aliasing on downscaling)
        frame_deg   = cv2.resize(frame_deg,   (TARGET_W, TARGET_H),
                                 interpolation=cv2.INTER_AREA)
        frame_clean = cv2.resize(frame_clean, (TARGET_W, TARGET_H),
                                 interpolation=cv2.INTER_AREA)

        # BGR → RGB for consistency with PyTorch tensor convention
        frame_deg   = cv2.cvtColor(frame_deg,   cv2.COLOR_BGR2RGB)
        frame_clean = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)

        # Transmission map: extract R channel directly to avoid chroma
        # averaging artefacts introduced by lossy video compression
        frame_t_resized = cv2.resize(frame_t, (TARGET_W, TARGET_H),
                                     interpolation=cv2.INTER_AREA)
        frame_t_gray    = frame_t_resized[:, :, 0]

        arr_deg[idx]   = frame_deg
        arr_clean[idx] = frame_clean
        arr_t[idx]     = frame_t_gray

        idx      += 1
        total_ok += 1

    cap_deg.release()
    cap_t.release()
    cap_clean.release()

    frames_per_video.append(total_ok)
    video_ids_used.append(seq)
    print(f"  Video {seq}: {total_ok} frames stored.")

# =============================================================================
# Step 5 — Trim preallocated arrays to actual frame count
# =============================================================================

arr_deg   = arr_deg[:idx]
arr_clean = arr_clean[:idx]
arr_t     = arr_t[:idx]

print(f"\nFinal array shapes:")
print(f"  arr_deg   : {arr_deg.shape}")
print(f"  arr_clean : {arr_clean.shape}")
print(f"  arr_t     : {arr_t.shape}")
print(f"  Videos processed : {len(frames_per_video)}")
print(f"  Total frames     : {idx}")
print(f"  Frames per video : {frames_per_video}")

# =============================================================================
# Step 6 — Sanity checks
# =============================================================================

assert arr_deg.shape[0] == arr_clean.shape[0] == arr_t.shape[0], \
    "Frame count mismatch across arrays."
assert arr_deg.shape[0] == sum(frames_per_video), \
    "frames_per_video does not sum to total frame count."
assert arr_deg.shape[1:3] == (TARGET_H, TARGET_W), \
    "Spatial resolution mismatch in degraded array."
assert arr_t.min() >= 0 and arr_t.max() <= 255, \
    "Transmission values out of uint8 range."

print(f"\nDataset statistics (uint8):")
print(f"  Degraded  — min: {arr_deg.min():3d}, max: {arr_deg.max():3d}, "
      f"mean: {arr_deg.mean():.1f}")
print(f"  Clean     — min: {arr_clean.min():3d}, max: {arr_clean.max():3d}, "
      f"mean: {arr_clean.mean():.1f}")
print(f"  Trans     — min: {arr_t.min():3d}, max: {arr_t.max():3d}, "
      f"mean: {arr_t.mean():.1f}")

# =============================================================================
# Step 7 — Save
# =============================================================================

np.save(os.path.join(out_path, "laparo_images_deg_x.npy"),   arr_deg)
np.save(os.path.join(out_path, "laparo_images_clean_j.npy"), arr_clean)
np.save(os.path.join(out_path, "trans_funcs_t.npy"),         arr_t)
np.save(os.path.join(out_path, "frames_per_video_used.npy"),
        np.array(frames_per_video, dtype=np.int64))
np.save(os.path.join(out_path, "video_ids_used.npy"),
        np.array(video_ids_used,   dtype=np.int64))

print(f"\nDataset construction complete.")
print(f"Output saved to : {out_path}")
print(f"Resolution      : {TARGET_H}×{TARGET_W}")
