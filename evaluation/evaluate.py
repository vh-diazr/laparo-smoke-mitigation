"""
evaluate.py
===========
Inference and quantitative evaluation of the hybrid physical-neural model
for nonhomogeneous smoke mitigation in laparoscopic images.

Given a degraded laparoscopic video sequence, the script:
  1. Loads the pre-trained HybridCNNTransformer model.
  2. Performs frame-level inference to estimate the transmission map T̂(x,y)
     and the per-channel ambient light vector Â.
  3. Recovers the smoke-free radiance field via atmospheric scattering inversion:
         Ĵ(x,y) = (f(x,y) - Â) / max(T̂(x,y), t_min) + Â
  4. Applies guided filter post-processing to the estimated transmission map
     to enforce local spatial coherence with the scene structure.
  5. Writes the restored video and estimated transmission map as output files.
  6. When a ground-truth clean reference sequence is provided, computes
     per-frame PSNR and SSIM metrics and aggregates mean ± std statistics.

Reference
---------
  Diaz-Ramirez et al., "Nonhomogeneous smoke mitigation in laparoscopic
  images using a hybrid physical-neural model", Medical & Biological
  Engineering & Computing, Springer Nature, 2025. (Under review.)

Usage
-----
  # Qualitative evaluation (no ground truth required):
  python evaluate.py --degraded path/to/degraded.mp4 \
                     --weights  path/to/model_weights.pth \
                     --output   path/to/output_dir/

  # Quantitative evaluation (with ground-truth reference):
  python evaluate.py --degraded      path/to/degraded.mp4 \
                     --ground_truth  path/to/clean.mp4    \
                     --weights       path/to/model_weights.pth \
                     --output        path/to/output_dir/
"""

import os
import csv
import time
import argparse

import cv2
import numpy as np
import torch

from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Model import — assumes evaluate.py is called from the repository root,
# so model/ is on the Python path via sys.path manipulation below.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from hybrid_cnn_vit import HybridCNNTransformer


# ===========================================================================
# 1. Device selection
# ===========================================================================

def get_device() -> torch.device:
    """
    Select the optimal available compute device.
    Priority: CUDA > Apple MPS > CPU.

    MPS (Metal Performance Shaders) is supported for Apple Silicon
    inference. CUDA is preferred on Linux/Windows GPU workstations.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# 2. Image quality metrics
# ===========================================================================

def compute_psnr(img_est: np.ndarray, img_ref: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two uint8 BGR images.

    PSNR = 10 · log₁₀(MAX² / MSE),  MAX = 255

    Parameters
    ----------
    img_est : np.ndarray, shape (H, W, 3), dtype uint8
        Estimated (restored) image.
    img_ref : np.ndarray, shape (H, W, 3), dtype uint8
        Ground-truth reference image.

    Returns
    -------
    float
        PSNR value in decibels. Returns inf if MSE = 0.
    """
    diff = img_est.astype(np.float64) - img_ref.astype(np.float64)
    mse  = np.mean(diff ** 2)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(img_est: np.ndarray, img_ref: np.ndarray) -> float:
    """
    Compute Structural Similarity Index Measure (SSIM) between two uint8
    BGR images.

    SSIM is evaluated on the grayscale luminance channel to provide a
    perceptually meaningful, single-valued similarity estimate consistent
    with the evaluation protocol reported in the manuscript.

    Parameters
    ----------
    img_est : np.ndarray, shape (H, W, 3), dtype uint8
        Estimated (restored) image.
    img_ref : np.ndarray, shape (H, W, 3), dtype uint8
        Ground-truth reference image.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """
    gray_est = cv2.cvtColor(img_est, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    return float(ssim_fn(gray_est, gray_ref, data_range=255))


# ===========================================================================
# 3. Physical-model-based restoration
# ===========================================================================

def restore_frame(
    img_bgr    : np.ndarray,
    model      : torch.nn.Module,
    device     : torch.device,
    train_h    : int,
    train_w    : int,
    gf_radius  : int   = 12,
    gf_eps     : float = 1e-2,
    t_min      : float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply the hybrid physical-neural restoration model to a single BGR frame.

    The restoration pipeline proceeds as follows:
      (i)   The input frame is resized to the training resolution and
            converted to a normalized RGB tensor.
      (ii)  The model estimates the transmission map T̂ ∈ [0,1]^{H×W} and
            the per-channel ambient light vector Â ∈ [0,1]³.
      (iii) T̂ is upsampled to the original frame resolution and refined
            using guided image filtering to enforce local spatial coherence
            with the scene structure.
      (iv)  The smoke-free radiance field is recovered via:
                Ĵ = (f - Â) / max(T̂, t_min) + Â
            where t_min prevents excessive amplification in near-zero
            transmission regions.

    Parameters
    ----------
    img_bgr   : np.ndarray, shape (H, W, 3), dtype uint8
        Input degraded BGR frame.
    model     : torch.nn.Module
        Pre-trained HybridCNNTransformer in evaluation mode.
    device    : torch.device
        Compute device.
    train_h   : int
        Training input height (pixels).
    train_w   : int
        Training input width (pixels).
    gf_radius : int
        Guided filter spatial radius (default: 12).
    gf_eps    : float
        Guided filter regularization parameter (default: 1e-2).
    t_min     : float
        Minimum transmission lower bound for numerical stability (default: 0.05).

    Returns
    -------
    restored_bgr : np.ndarray, shape (H, W, 3), dtype uint8
        Restored smoke-free frame in BGR format.
    trans_map    : np.ndarray, shape (H, W), dtype float32
        Refined transmission map T̂(x,y) at original resolution, values in [t_min, 1].
    A_scalar     : float
        Estimated ambient light scalar Â in [0, 255]. The A_head produces a
        single wavelength-independent value enforcing A_R = A_G = A_B,
        consistent with the scalar airlight assumption used during dataset
        construction (apply_smoke_degradation.py).
    """
    H_orig, W_orig = img_bgr.shape[:2]

    # (i) Preprocessing
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (train_w, train_h), interpolation=cv2.INTER_AREA)
    tensor      = (torch.from_numpy(img_resized)
                       .permute(2, 0, 1)
                       .unsqueeze(0)
                       .float()
                       .div(255.0)
                       .to(device))

    # (ii) Network forward pass
    with torch.no_grad():
        T_pred, A_pred = model(tensor)

    # (iii) Transmission map post-processing
    trans_small = T_pred[0, 0].cpu().numpy().astype(np.float32)
    trans_full  = cv2.resize(trans_small, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    trans_full  = cv2.ximgproc.guidedFilter(
        guide = img_bgr,
        src   = trans_full,
        radius= gf_radius,
        eps   = gf_eps,
    )
    trans_full = np.clip(trans_full, t_min, 1.0)

    # Ambient light scalar Â ∈ [0,1] → scaled to [0,255]
    # The A_head outputs a single wavelength-independent scalar enforcing the
    # assumption A_R = A_G = A_B, consistent with the dataset construction
    # protocol in apply_smoke_degradation.py (scalar airlight parameter).
    # The model forward pass expands this scalar to (B,3) via A.expand(-1,3).
    A_scalar = float(A_pred[0, 0].cpu().item()) * 255.0

    # (iv) Atmospheric scattering inversion (wavelength-independent airlight)
    #   Ĵ(x,y) = (f(x,y) - Â) / max(T̂(x,y), t_min) + Â
    I_f          = img_bgr.astype(np.float32)
    T_f          = trans_full[:, :, np.newaxis]     # broadcast over channels
    restored     = (I_f - A_scalar) / T_f + A_scalar
    restored_bgr = np.clip(restored, 0.0, 255.0).astype(np.uint8)

    return restored_bgr, trans_full, A_scalar


# ===========================================================================
# 4. Main evaluation loop
# ===========================================================================

def evaluate(args: argparse.Namespace) -> None:
    """
    Run inference on a degraded laparoscopic video sequence and, optionally,
    compute reference-based quality metrics (PSNR, SSIM).
    """
    os.makedirs(args.output, exist_ok=True)

    # -----------------------------------------------------------------------
    # Device and model initialization
    # -----------------------------------------------------------------------
    device = get_device()
    print(f"Compute device: {device}")

    model = HybridCNNTransformer(
        img_hw    = (args.train_h, args.train_w),
        embed_dim = args.embed_dim,
        num_layers= args.num_layers,
        num_heads = args.num_heads,
    ).to(device)

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model weights loaded from: {args.weights}")

    # -----------------------------------------------------------------------
    # Input video capture
    # -----------------------------------------------------------------------
    cap_deg = cv2.VideoCapture(args.degraded)
    if not cap_deg.isOpened():
        raise FileNotFoundError(f"Cannot open degraded sequence: {args.degraded}")

    n_frames = int(cap_deg.get(cv2.CAP_PROP_FRAME_COUNT))
    W        = int(cap_deg.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap_deg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in   = cap_deg.get(cv2.CAP_PROP_FPS) or 25.0

    print(f"Input sequence  : {args.degraded}")
    print(f"Resolution      : {H} × {W}")
    print(f"Frames          : {n_frames}")

    # Optional ground-truth capture
    cap_gt      = None
    gt_available= False
    if args.ground_truth and os.path.exists(args.ground_truth):
        cap_gt       = cv2.VideoCapture(args.ground_truth)
        gt_available = cap_gt.isOpened()
        print(f"Ground truth    : {args.ground_truth}  (quantitative metrics enabled)")
    else:
        print("Ground truth    : not provided (qualitative evaluation only)")

    # -----------------------------------------------------------------------
    # Output video writers
    # -----------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_restored = cv2.VideoWriter(
        os.path.join(args.output, "restored.mp4"), fourcc, fps_in, (W, H)
    )
    # Transmission map visualized as a single-channel heat map (COLORMAP_JET)
    writer_trans = cv2.VideoWriter(
        os.path.join(args.output, "transmission.mp4"), fourcc, fps_in, (W, H)
    )

    # -----------------------------------------------------------------------
    # Per-frame processing
    # -----------------------------------------------------------------------
    psnr_list, ssim_list, time_list = [], [], []

    for n in range(n_frames):
        ret_deg, frame_deg = cap_deg.read()
        if not ret_deg:
            break

        frame_gt = None
        if gt_available:
            ret_gt, frame_gt = cap_gt.read()
            if not ret_gt:
                gt_available = False
                frame_gt     = None

        # Restoration
        t0 = time.perf_counter()
        restored, trans_map, A_val = restore_frame(
            img_bgr   = frame_deg,
            model     = model,
            device    = device,
            train_h   = args.train_h,
            train_w   = args.train_w,
            gf_radius = args.gf_radius,
            gf_eps    = args.gf_eps,
            t_min     = args.t_min,
        )
        time_list.append(time.perf_counter() - t0)

        # Write outputs
        writer_restored.write(restored)
        trans_vis = cv2.applyColorMap(
            (trans_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        writer_trans.write(trans_vis)

        # Quality metrics
        if frame_gt is not None:
            psnr_list.append(compute_psnr(restored, frame_gt))
            ssim_list.append(compute_ssim(restored, frame_gt))

        if (n + 1) % 50 == 0 or n == 0:
            print(
                f"  Frame {n+1:4d}/{n_frames} | "
                f"A = {A_val:.1f} | "
                f"T ∈ [{trans_map.min():.3f}, {trans_map.max():.3f}] | "
                f"t = {time_list[-1]*1e3:.1f} ms"
            )

    # -----------------------------------------------------------------------
    # Release resources
    # -----------------------------------------------------------------------
    cap_deg.release()
    if cap_gt is not None:
        cap_gt.release()
    writer_restored.release()
    writer_trans.release()

    # -----------------------------------------------------------------------
    # Quantitative results
    # -----------------------------------------------------------------------
    summary_lines = []

    if psnr_list:
        psnr_arr = np.array(psnr_list)
        ssim_arr = np.array(ssim_list)
        time_arr = np.array(time_list)

        summary_lines += [
            f"Frames evaluated : {len(psnr_arr)}",
            f"PSNR (dB)        : {psnr_arr.mean():.4f} ± {psnr_arr.std():.4f}",
            f"SSIM             : {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}",
            f"Inference time   : {time_arr.mean()*1e3:.2f} ± {time_arr.std()*1e3:.2f} ms/frame",
            f"Throughput       : {1.0/time_arr.mean():.2f} fps",
        ]

        # Write per-frame CSV
        csv_path = os.path.join(args.output, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["frame", "psnr_db", "ssim", "time_ms"])
            writer_csv.writeheader()
            for i, (p, s, t) in enumerate(zip(psnr_list, ssim_list, time_list)):
                writer_csv.writerow({
                    "frame"   : i + 1,
                    "psnr_db" : round(p, 6),
                    "ssim"    : round(s, 6),
                    "time_ms" : round(t * 1e3, 4),
                })
        print(f"\nPer-frame metrics saved to: {csv_path}")
    else:
        t_arr = np.array(time_list)
        summary_lines += [
            f"Frames processed : {len(t_arr)}",
            f"Inference time   : {t_arr.mean()*1e3:.2f} ± {t_arr.std()*1e3:.2f} ms/frame",
            f"Throughput       : {1.0/t_arr.mean():.2f} fps",
            "(No ground-truth provided; PSNR/SSIM not computed.)",
        ]

    # Write summary
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n--- Evaluation Summary ---")
    print("\n".join(summary_lines))
    print(f"\nOutputs written to: {args.output}")


# ===========================================================================
# 5. Argument parser
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference and evaluation script for the hybrid physical-neural "
            "model for nonhomogeneous smoke mitigation in laparoscopic images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--degraded",      required=True,
                        help="Path to the degraded input video sequence (.mp4).")
    parser.add_argument("--ground_truth",  default=None,
                        help="Path to the ground-truth clean video sequence (.mp4). "
                             "If provided, PSNR and SSIM are computed.")
    parser.add_argument("--weights",       required=True,
                        help="Path to the pre-trained model weights (.pth).")
    parser.add_argument("--output",        default="results/",
                        help="Directory for output files.")

    # Model hyperparameters (must match training configuration)
    parser.add_argument("--train_h",    type=int,   default=480,
                        help="Training input height (pixels).")
    parser.add_argument("--train_w",    type=int,   default=512,
                        help="Training input width (pixels).")
    parser.add_argument("--embed_dim",  type=int,   default=256,
                        help="Transformer embedding dimension.")
    parser.add_argument("--num_layers", type=int,   default=4,
                        help="Number of Transformer encoder layers.")
    parser.add_argument("--num_heads",  type=int,   default=8,
                        help="Number of multi-head self-attention heads.")

    # Restoration hyperparameters
    parser.add_argument("--gf_radius",  type=int,   default=12,
                        help="Guided filter spatial radius.")
    parser.add_argument("--gf_eps",     type=float, default=1e-2,
                        help="Guided filter regularization parameter.")
    parser.add_argument("--t_min",      type=float, default=0.05,
                        help="Minimum transmission lower bound.")

    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    evaluate(parse_args())
