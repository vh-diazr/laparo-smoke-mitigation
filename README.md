# Nonhomogeneous Smoke Mitigation in Laparoscopic Images Using a Hybrid Physical-Neural Model

**Manuscript under minor revision** — *Medical & Biological Engineering & Computing* (MBEC), Springer Nature.

**Authors:** Victor Diaz-Ramirez¹, Jose Godoy¹, Christian Gaxiola¹, Rigoberto Juarez-Salazar¹, Leonardo Trujillo²

¹ Instituto Politécnico Nacional – CITEDI, Tijuana, Mexico
² Tecnológico Nacional de México / IT Tijuana, Tijuana, Mexico

This repository provides the model architecture, physics-based smoke simulation, training pipeline, and evaluation code associated with the manuscript. Representative degraded laparoscopic sequences are included under `evaluation/samples/` to enable immediate inference without requiring access to the full dataset.

---

## Repository Structure

```
laparo-smoke-mitigation/
│
├── README.md
├── model/
│   └── hybrid_cnn_vit.py          ← Hybrid CNN-ViT architecture
├── simulation/
│   └── smoke_simulation.py        ← Navier-Stokes surgical smoke simulation
├── dataset/
│   └── build_dataset.py           ← Synthetic dataset construction pipeline
├── training/
│   └── train.py                   ← Model training script
├── evaluation/
│   ├── evaluate.py                ← Inference and quantitative evaluation
│   └── samples/
│       ├── seq_01/                ← Synthetic smoke: dense localized distribution
│       ├── seq_02/                ← Synthetic smoke: diffuse spatial distribution
│       └── seq_03/                ← Real electrocautery smoke (Cholec80)
└── weights/
    └── model_weights.pth          ← Pre-trained model weights
```

---

## Method

### Physical Image Formation Model

Laparoscopic smoke degradation is modelled via the atmospheric scattering equation:

```
f(x,y) = J(x,y) · T(x,y) + A · (1 - T(x,y))
```

where `f(x,y)` is the degraded observation, `J(x,y)` is the latent smoke-free radiance field, `T(x,y) ∈ [0,1]` is the spatially varying transmission map, and `A ∈ ℝ³` is the per-channel ambient light vector. Smoke-free image recovery is performed by inverting this model given the network estimates `(T̂, Â)`:

```
Ĵ(x,y) = (f(x,y) - Â) / max(T̂(x,y), ε) + Â,   ε = 0.05
```

### Hybrid CNN-ViT Architecture

The model jointly estimates `T̂` and `Â` from the degraded input frame. The CNN encoder extracts hierarchical local features via four strided convolutional stages (channel widths: 32→64→128→`embed_dim`), progressively downsampling spatial resolution by a factor of 16. The resulting feature map is flattened into a sequence of tokens and processed by a Vision Transformer (ViT) encoder comprising `L` layers of multi-head self-attention (MHSA) with learned positional embeddings, capturing long-range spatial dependencies across the full spatial extent of the degraded image. The CNN decoder reconstructs the full-resolution transmission map via transposed convolutions with skip connections from the encoder. A parallel ambient light estimation head applies global average pooling over the bottleneck feature map followed by a 1×1 convolution and sigmoid activation, producing per-channel estimates `Â ∈ [0,1]³`.

Key architectural parameters: `embed_dim = 128`, `num_layers = 2`, `num_heads = 8`, input resolution `480 × 512`.

---

## Dataset Access

The full training and evaluation dataset (~8 GB) is hosted on Google Drive. Access is provided upon request for legitimate research use.

**Send a request to [vdiazr@ipn.mx](mailto:vdiazr@ipn.mx) with subject:**
```
[Dataset Request] laparo-smoke-mitigation
```
Include your full name, institutional affiliation, and intended use. A download link and decompression password will be provided within 3 business days.

| Property | Value |
|---|---|
| Training sequences | 30 laparoscopic videos (EndoScapes) |
| Training triplets | 9,460 `{f(x,y), J(x,y), T(x,y)}` |
| Evaluation sequences | 10 laparoscopic sequences (Cholec80) |
| Evaluation frames | 1,981 synthetically degraded frames |
| Image resolution | 512 × 480 pixels |

---

## Installation

```bash
git clone https://github.com/[username]/laparo-smoke-mitigation.git
cd laparo-smoke-mitigation

python3 -m venv env
source env/bin/activate

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu125
pip install numpy==2.4.1 opencv-python==4.13.0 opencv-contrib-python==4.13.0 scikit-image
```

> **Note:** `opencv-contrib-python` is required for the guided filter post-processing step applied to the estimated transmission map.

---

## Evaluation

The `evaluate.py` script performs frame-level inference on a degraded video sequence, applies the physics-based restoration model, and — when a ground-truth reference sequence is provided — computes per-frame and aggregate PSNR and SSIM metrics.

### Usage

```bash
# Qualitative evaluation only (no ground truth required)
python evaluation/evaluate.py \
    --degraded evaluation/samples/seq_01/degraded.mp4 \
    --weights  weights/model_weights.pth \
    --output   results/seq_01/

# Quantitative evaluation (with ground-truth reference)
python evaluation/evaluate.py \
    --degraded    evaluation/samples/seq_01/degraded.mp4 \
    --ground_truth evaluation/samples/seq_01/clean.mp4 \
    --weights     weights/model_weights.pth \
    --output      results/seq_01/
```

### Outputs

| File | Description |
|---|---|
| `restored.mp4` | Restored video sequence |
| `transmission.mp4` | Estimated transmission map `T̂(x,y,t)` |
| `metrics.csv` | Per-frame PSNR and SSIM (when ground truth is provided) |
| `summary.txt` | Aggregate mean ± std statistics |

---

## Computational Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| OS | Debian GNU/Linux 13 |
| Python | 3.12.3 |
| PyTorch | 2.5.1 |
| CUDA | 12.5 |
| NumPy | 2.4.1 |
| OpenCV | 4.13.0 |

---

## Citation

```bibtex
@article{diazramirez2025nonhomogeneous,
  title     = {Nonhomogeneous smoke mitigation in laparoscopic images
               using a hybrid physical-neural model},
  author    = {Diaz-Ramirez, Victor and Godoy, Jose and Gaxiola, Christian
               and Juarez-Salazar, Rigoberto and Trujillo, Leonardo},
  journal   = {Medical {\&} Biological Engineering {\&} Computing},
  publisher = {Springer Nature},
  year      = {2025},
  note      = {Under review}
}
```

---

## Funding

Instituto Politécnico Nacional (project **SIP-20263728**) and Secretaría de Ciencia, Humanidades, Tecnología e Innovación (**Secihti**).

---

## Contact

**Victor Diaz-Ramirez** (Corresponding Author) — vdiazr@ipn.mx
Instituto Politécnico Nacional – CITEDI, Tijuana, Mexico
