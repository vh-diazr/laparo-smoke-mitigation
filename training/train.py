"""
train.py
========
Full training script for the hybrid physical-neural model for
nonhomogeneous smoke mitigation in laparoscopic images.

Scientific context
------------------
The model is trained to jointly estimate the spatially varying transmission
map T(x,y) and the per-channel ambient light vector A from synthetically
smoke-degraded laparoscopic frames. Image restoration is performed via
explicit inversion of the atmospheric scattering equation:

    J(x,y) = (f(x,y) - A) / max(T(x,y), eps) + A

The composite training loss is:

    L = (1 - lambda_T) * L_restoration + lambda_T * L_transmission

where L_restoration = l1(J_pred, J_gt) and L_transmission = l1(T_pred, T_gt).
The weighting coefficient lambda_T is annealed via a cosine schedule from
lambda_T_start = 0.7 to lambda_T_end = 0.3, progressively shifting
optimization emphasis from transmission estimation toward perceptual
restoration quality.

Training protocol
-----------------
  - Optimizer       : AdamW (lr = 1e-4, weight_decay = 1e-4)
  - LR scheduler    : Linear warmup (5 epochs) + cosine annealing to 1e-6
  - Batch size      : 12
  - Max epochs      : 100
  - Early stopping  : patience = 20 epochs on validation loss
  - Gradient clip   : max_norm = 1.0
  - Data split      : video-level 90/10 train/val (prevents temporal leakage)
  - Augmentation    : 180° rotation + horizontal flip + vertical flip (train only)
  - Input resolution: 480 × 512 pixels

Pipeline position
-----------------
  construct_dataset.py
          ↓
  {laparo_images_deg_x.npy,
   laparo_images_clean_j.npy,
   trans_funcs_t.npy,
   frames_per_video_used.npy}
          ↓
  train.py
          ↓
  best_full.pth  ← optimal model weights (minimum validation loss)
  last_full.pth  ← final model weights (last epoch)

Prerequisites
-------------
  pip install torch torchvision numpy

Reference
---------
  Diaz-Ramirez et al., "Nonhomogeneous smoke mitigation in laparoscopic
  images using a hybrid physical-neural model", Medical & Biological
  Engineering & Computing, Springer Nature, 2025. (Under review.)

Usage
-----
  1. Set base path below to match your directory structure.
  2. Run: python train.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from hybrid_cnn_vit import HybridCNNTransformer


# =============================================================================
# 1. Dataset with online augmentation
# =============================================================================

class LaparoMemmapDataset(Dataset):
    """
    Memory-mapped dataset for laparoscopic smoke degradation triplets.

    Loads samples lazily from .npy arrays using memory-mapping, avoiding
    full dataset loading into RAM. Each sample is a triplet:
      xb — degraded frame f(x,y),          shape (3, H, W), float32 in [0,1]
      tb — ground-truth transmission map T, shape (1, H, W), float32 in [0,1]
      jb — ground-truth clean frame J(x,y), shape (3, H, W), float32 in [0,1]

    Online augmentation (training only)
    ------------------------------------
    Three independent random geometric transforms are applied consistently
    to all three fields of each triplet:
      - Random 180° rotation (preserves H×W for non-square 480×512 inputs)
      - Random horizontal flip
      - Random vertical flip
    This produces up to 6 geometrically distinct variants per frame without
    any disk overhead.

    Note: 90° and 270° rotations are excluded because they swap H and W
    (480×512 → 512×480), which is incompatible with the fixed-resolution
    CNN encoder architecture.
    """

    def __init__(self, x_path: str, j_path: str, t_path: str,
                 augment: bool = False):
        self.x       = np.load(x_path, mmap_mode="r")
        self.j       = np.load(j_path, mmap_mode="r")
        self.t       = np.load(t_path, mmap_mode="r")
        self.augment = augment

        assert self.x.shape[0] == self.j.shape[0] == self.t.shape[0], \
            "Frame count mismatch across dataset arrays."
        assert self.x.ndim == 4 and self.x.shape[-1] == 3, \
            "Degraded array must be (N, H, W, 3)."
        assert self.j.ndim == 4 and self.j.shape[-1] == 3, \
            "Clean array must be (N, H, W, 3)."
        assert self.t.ndim in (3, 4), \
            "Transmission array must be (N, H, W) or (N, H, W, 1)."

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        xb = torch.from_numpy(self.x[idx].copy()).permute(2, 0, 1).contiguous()
        jb = torch.from_numpy(self.j[idx].copy()).permute(2, 0, 1).contiguous()

        tb = self.t[idx].copy()
        if tb.ndim == 2:
            tb = torch.from_numpy(tb).unsqueeze(0)
        elif tb.ndim == 3:
            tb = torch.from_numpy(tb).permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected transmission map shape: {tb.shape}")

        # Normalize to [0, 1]
        xb = xb.float() / 255.0
        jb = jb.float() / 255.0
        tb = tb.float() / 255.0

        # Online augmentation (training only)
        if self.augment:
            if random.random() > 0.5:                      # 180° rotation
                xb = torch.rot90(xb, 2, dims=(1, 2))
                jb = torch.rot90(jb, 2, dims=(1, 2))
                tb = torch.rot90(tb, 2, dims=(1, 2))
            if random.random() > 0.5:                      # horizontal flip
                xb = torch.flip(xb, dims=(2,))
                jb = torch.flip(jb, dims=(2,))
                tb = torch.flip(tb, dims=(2,))
            if random.random() > 0.5:                      # vertical flip
                xb = torch.flip(xb, dims=(1,))
                jb = torch.flip(jb, dims=(1,))
                tb = torch.flip(tb, dims=(1,))

        return xb, tb, jb


# =============================================================================
# 2. Utilities
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def restore_image(I_deg: torch.Tensor, T: torch.Tensor, A: torch.Tensor,
                  eps: float = 1e-3, clamp: bool = True) -> torch.Tensor:
    """
    Recover the smoke-free radiance field via atmospheric scattering inversion:

        J = (I - A) / max(T, eps) + A

    Parameters
    ----------
    I_deg : (B, 3, H, W) — degraded input frames
    T     : (B, 1, H, W) — estimated transmission maps
    A     : (B, 3) or (B, 3, 1, 1) — estimated ambient light vectors
    eps   : float — minimum transmission lower bound for numerical stability
    clamp : bool  — clamp output to [0, 1]
    """
    A = A.to(device=I_deg.device, dtype=I_deg.dtype)
    T = T.to(device=I_deg.device, dtype=I_deg.dtype)
    if A.dim() == 2:
        A = A[:, :, None, None]
    J = (I_deg - A) / T.clamp_min(eps) + A
    return J.clamp(0.0, 1.0) if clamp else J


# =============================================================================
# 3. Lambda_T cosine annealing schedule
# =============================================================================

def get_lambda_T(epoch: int, Nepochs: int,
                 lambda_T_start: float = 0.7,
                 lambda_T_end:   float = 0.3) -> float:
    """
    Cosine annealing schedule for the transmission loss weighting coefficient.

    lambda_T decays from lambda_T_start to lambda_T_end over training,
    progressively shifting optimization emphasis from accurate transmission
    map estimation toward perceptual image restoration quality.

    Parameters
    ----------
    epoch          : current epoch index (0-indexed)
    Nepochs        : total number of training epochs
    lambda_T_start : initial transmission loss weight (default: 0.7)
    lambda_T_end   : final transmission loss weight (default: 0.3)
    """
    progress = epoch / max(1, Nepochs - 1)
    return lambda_T_end + 0.5 * (lambda_T_start - lambda_T_end) * (
        1.0 + math.cos(math.pi * progress)
    )


# =============================================================================
# 4. Learning rate scheduler
# =============================================================================

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup for warmup_epochs, followed by cosine annealing to eta_min.

    This combined schedule stabilizes early training by gradually increasing
    the learning rate during warmup, then provides smooth decay to prevent
    oscillation near convergence.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]


# =============================================================================
# 5. Video-level train/val split
# =============================================================================

def build_video_ranges(frames_per_video: np.ndarray) -> list:
    """Build (start, end) frame index ranges for each video."""
    ranges, start = [], 0
    for nf in frames_per_video:
        end = start + int(nf)
        ranges.append((start, end))
        start = end
    return ranges


def videos_to_frame_indices(video_ids: list,
                             video_ranges: list) -> np.ndarray:
    """Expand a list of video indices to the corresponding frame indices."""
    idx = [np.arange(*video_ranges[v], dtype=np.int64) for v in video_ids]
    return np.concatenate(idx) if idx else np.array([], dtype=np.int64)


def make_video_split(frames_per_video: np.ndarray,
                     val_fraction: float = 0.10,
                     seed: int = 42):
    """
    Partition videos into training and validation sets at the video level,
    then expand to frame indices. Video-level partitioning prevents temporal
    data leakage between adjacent frames of the same sequence.

    Parameters
    ----------
    frames_per_video : array of per-video frame counts
    val_fraction     : fraction of videos allocated to validation (default: 0.10)
    seed             : random seed for reproducibility

    Returns
    -------
    train_idx  : frame indices for training
    val_idx    : frame indices for validation
    train_vids : video indices assigned to training
    val_vids   : video indices assigned to validation
    """
    n_vids = len(frames_per_video)
    vids   = np.arange(n_vids)
    rng    = np.random.default_rng(seed)
    rng.shuffle(vids)

    n_val      = max(1, int(round(val_fraction * n_vids)))
    val_vids   = vids[:n_val].tolist()
    train_vids = vids[n_val:].tolist()

    video_ranges = build_video_ranges(frames_per_video)
    train_idx    = videos_to_frame_indices(train_vids, video_ranges)
    val_idx      = videos_to_frame_indices(val_vids,   video_ranges)

    return train_idx, val_idx, train_vids, val_vids


# =============================================================================
# 6. Training loop
# =============================================================================

def model_training(
    model, train_dl, val_dl, optimizer, scheduler, device, Nepochs,
    patience       : int   = 20,
    min_delta      : float = 1e-4,
    best_path      : str   = "best_full.pth",
    grad_clip      : float = 1.0,
    lambda_T_start : float = 0.7,
    lambda_T_end   : float = 0.3,
    eps_rest       : float = 1e-3,
    k_trans        : float = 10.0,
) -> None:
    """
    Training loop with early stopping on validation loss.

    At each epoch:
      1. Computes the annealed lambda_T for the composite loss weighting.
      2. Runs the training pass with gradient clipping.
      3. Runs the validation pass without gradient computation.
      4. Saves the model state when validation loss improves.
      5. Terminates early if no improvement for `patience` consecutive epochs.

    Parameters
    ----------
    model          : HybridCNNTransformer instance
    train_dl       : training DataLoader (augmented)
    val_dl         : validation DataLoader (no augmentation)
    optimizer      : AdamW optimizer
    scheduler      : WarmupCosineScheduler instance
    device         : torch.device
    Nepochs        : maximum number of training epochs
    patience       : early stopping patience in epochs
    min_delta      : minimum validation loss improvement to reset patience
    best_path      : output path for best model weights (.pth)
    grad_clip      : gradient clipping max norm
    lambda_T_start : initial transmission loss weight
    lambda_T_end   : final transmission loss weight
    eps_rest       : minimum transmission lower bound for restoration
    k_trans        : transmission accuracy score scaling factor
    """
    best_val   = float("inf")
    bad_epochs = 0

    for epoch in range(Nepochs):
        lambda_T   = get_lambda_T(epoch, Nepochs, lambda_T_start, lambda_T_end)
        current_lr = optimizer.param_groups[0]['lr']

        # --- Training pass ---
        model.train()
        train_loss_sum = 0.0
        train_accT_sum = 0.0
        train_psnr_sum = 0.0
        n_train        = 0

        for xb, tb, jb in train_dl:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            jb = jb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            T_pred, A_pred = model(xb)
            if A_pred.dim() == 2:
                A_pred = A_pred[:, :, None, None]
            if T_pred.shape[-2:] != tb.shape[-2:]:
                T_pred = F.interpolate(T_pred, size=tb.shape[-2:],
                                       mode="bilinear", align_corners=False)

            J_pred = restore_image(xb, T_pred, A_pred, eps=eps_rest, clamp=True)

            loss_img = F.l1_loss(J_pred, jb)
            loss_T   = F.l1_loss(T_pred, tb)
            loss     = (1.0 - lambda_T) * loss_img + lambda_T * loss_T

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=grad_clip)
            optimizer.step()

            bs             = xb.size(0)
            train_loss_sum += loss.item() * bs
            n_train        += bs

            with torch.no_grad():
                nmae_T        = (T_pred - tb).abs().mean(dim=(1, 2, 3))
                score_T       = torch.exp(-k_trans * nmae_T)
                train_accT_sum += score_T.mean().item() * bs

                mse            = (J_pred - jb).pow(2).mean(dim=(1, 2, 3))
                psnr           = 10.0 * torch.log10(1.0 / (mse + 1e-10))
                train_psnr_sum += psnr.mean().item() * bs

        train_loss = train_loss_sum / max(1, n_train)
        train_accT = train_accT_sum / max(1, n_train)
        train_psnr = train_psnr_sum / max(1, n_train)

        # --- Validation pass ---
        model.eval()
        val_loss_sum = 0.0
        val_accT_sum = 0.0
        val_psnr_sum = 0.0
        n_val        = 0

        with torch.no_grad():
            for xb, tb, jb in val_dl:
                xb = xb.to(device, non_blocking=True)
                tb = tb.to(device, non_blocking=True)
                jb = jb.to(device, non_blocking=True)

                T_pred, A_pred = model(xb)
                if A_pred.dim() == 2:
                    A_pred = A_pred[:, :, None, None]
                if T_pred.shape[-2:] != tb.shape[-2:]:
                    T_pred = F.interpolate(T_pred, size=tb.shape[-2:],
                                           mode="bilinear", align_corners=False)

                J_pred = restore_image(xb, T_pred, A_pred, eps=eps_rest, clamp=True)

                loss_img     = F.l1_loss(J_pred, jb)
                loss_T       = F.l1_loss(T_pred, tb)
                loss         = (1.0 - lambda_T) * loss_img + lambda_T * loss_T

                bs           = xb.size(0)
                val_loss_sum += loss.item() * bs
                n_val        += bs

                nmae_T       = (T_pred - tb).abs().mean(dim=(1, 2, 3))
                score_T      = torch.exp(-k_trans * nmae_T)
                val_accT_sum += score_T.mean().item() * bs

                mse          = (J_pred - jb).pow(2).mean(dim=(1, 2, 3))
                psnr         = 10.0 * torch.log10(1.0 / (mse + 1e-10))
                val_psnr_sum += psnr.mean().item() * bs

        val_loss = val_loss_sum / max(1, n_val)
        val_accT = val_accT_sum / max(1, n_val)
        val_psnr = val_psnr_sum / max(1, n_val)

        # --- Early stopping and model checkpoint ---
        if val_loss < best_val - min_delta:
            best_val   = val_loss
            bad_epochs = 0
            state      = (model.module.state_dict()
                          if hasattr(model, 'module') else model.state_dict())
            torch.save(state, best_path)
            tag = " (saved best)"
        else:
            bad_epochs += 1
            tag         = ""

        print(
            f"Epoch {epoch+1:03d}/{Nepochs:03d} | "
            f"lr {current_lr:.2e} | λ_T {lambda_T:.3f} | "
            f"train loss {train_loss:.6f} | train accT {train_accT*100:.2f}% | "
            f"train PSNR {train_psnr:.2f} dB || "
            f"val loss {val_loss:.6f} | val accT {val_accT*100:.2f}% | "
            f"val PSNR {val_psnr:.2f} dB | "
            f"ES {bad_epochs}/{patience}{tag}"
        )

        scheduler.step()

        if bad_epochs >= patience:
            print(f"Early stopping triggered. Best validation loss: {best_val:.6f}")
            break


# =============================================================================
# 7. Main
# =============================================================================

def main():
    # Set your path here
    base    = "/your/path/to/laparo_dehazing/"
    x_path  = os.path.join(base, "full_train_data", "laparo_images_deg_x.npy")
    j_path  = os.path.join(base, "full_train_data", "laparo_images_clean_j.npy")
    t_path  = os.path.join(base, "full_train_data", "trans_funcs_t.npy")
    fpv_path= os.path.join(base, "full_train_data", "frames_per_video_used.npy")

    out_dir   = os.path.join(base, "final_train")
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best_full.pth")
    last_path = os.path.join(out_dir, "last_full.pth")

    # --- Hyperparameters ---
    seed             = 42
    batch_size       = 12
    Nepochs          = 100
    val_fraction     = 0.10
    patience         = 20
    min_delta        = 1e-4
    grad_clip        = 1.0
    eps_rest         = 1e-3
    lr               = 1e-4
    wd               = 1e-4
    lambda_T_start   = 0.7
    lambda_T_end     = 0.3
    warmup_epochs    = 5
    num_workers      = 4
    pin_memory       = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    set_seed(seed)

    # --- Dataset ---
    dataset_train = LaparoMemmapDataset(x_path, j_path, t_path, augment=True)
    dataset_val   = LaparoMemmapDataset(x_path, j_path, t_path, augment=False)
    print(f"Total frames : {len(dataset_train)}")

    # Video-level train/val split
    frames_per_video = np.load(fpv_path)
    train_idx, val_idx, train_vids, val_vids = make_video_split(
        frames_per_video, val_fraction=val_fraction, seed=seed
    )

    assert np.intersect1d(train_idx, val_idx).size == 0, \
        "Train/validation frame index overlap detected."

    print(f"Train videos : {len(train_vids)} | Val videos : {len(val_vids)}")
    print(f"Train frames : {len(train_idx)} | Val frames : {len(val_idx)}")

    # --- DataLoaders ---
    train_dl = DataLoader(
        Subset(dataset_train, train_idx.tolist()),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_dl = DataLoader(
        Subset(dataset_val, val_idx.tolist()),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # --- Model ---
    model     = HybridCNNTransformer(img_hw=(480, 512)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=warmup_epochs,
        total_epochs=Nepochs, eta_min=1e-6,
    )

    print(f"Lambda_T schedule  : {lambda_T_start:.2f} → {lambda_T_end:.2f} (cosine)")
    print(f"LR schedule        : warmup {warmup_epochs} epochs → cosine annealing to 1e-6")
    print(f"Augmentation       : 180° rotation + H-flip + V-flip (train only)")

    # --- Train ---
    model_training(
        model=model, train_dl=train_dl, val_dl=val_dl,
        optimizer=optimizer, scheduler=scheduler,
        device=device, Nepochs=Nepochs,
        patience=patience, min_delta=min_delta,
        best_path=best_path, grad_clip=grad_clip,
        lambda_T_start=lambda_T_start, lambda_T_end=lambda_T_end,
        eps_rest=eps_rest, k_trans=10.0,
    )

    # Save final state
    state = (model.module.state_dict()
             if hasattr(model, 'module') else model.state_dict())
    torch.save(state, last_path)

    print(f"\nTraining complete.")
    print(f"  Best weights : {best_path}")
    print(f"  Last weights : {last_path}")


if __name__ == "__main__":
    main()
