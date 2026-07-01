import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Basic conv block used in encoder/decoder."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HybridCNNTransformer(nn.Module):
    """
    Hybrid model for dense regression (e.g., transmission-map estimation).

    Pipeline:
      CNN encoder (local features + downsampling)
        -> tokenization at 1/16 resolution
        -> Transformer encoder (global context across tokens)
        -> CNN decoder (upsampling to full resolution)
        -> 1-channel output (transmission map T)

    The atmospheric light head estimates a per-channel vector A ∈ [0,1]^3,
    allowing wavelength-dependent (R ≠ G ≠ B) or wavelength-independent
    (R = G = B) ambient light to be represented.  For applications that
    require a scalar ambient light parameter the caller may collapse the
    three estimates via an RGB-to-grayscale conversion (see `rgb_to_gray_A`).

    Input : (N, 3, H, W)  e.g. (N, 3, 480, 512)
    Output: tuple(T, A)
        T : (N, 1, H, W)  — transmission map, values in [0, 1]
        A : (N, 3)        — per-channel atmospheric light, values in [0, 1]
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 1,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        img_hw: tuple[int, int] = (480, 512),   # used to size positional embeddings
    ):
        super().__init__()

        H, W = img_hw
        assert H % 16 == 0 and W % 16 == 0, "img_hw must be divisible by 16 for this model."
        self.H16, self.W16 = H // 16, W // 16
        self.num_tokens = self.H16 * self.W16
        self.embed_dim = embed_dim

        # -------------------------
        # CNN encoder (downsample x16)
        # -------------------------
        # 480x512 -> 240x256 -> 120x128 -> 60x64 -> 30x32
        self.enc1 = nn.Sequential(ConvBNReLU(in_ch, 32), ConvBNReLU(32, 32))
        self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, bias=False)

        self.enc2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)

        self.enc3 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        self.down3 = nn.Conv2d(128, 128, kernel_size=2, stride=2, bias=False)

        self.enc4 = nn.Sequential(ConvBNReLU(128, embed_dim), ConvBNReLU(embed_dim, embed_dim))
        self.down4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=2, stride=2, bias=False)

        # -------------------------
        # Transformer encoder (global context)
        # -------------------------
        # Positional embedding for tokens at 1/16 resolution
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # (B, N, C)
            norm_first=True,
        )

        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.transformer.gradient_checkpointing = True  # or use torch.utils.checkpoint
        self.norm = nn.LayerNorm(embed_dim)

        # -------------------------
        # Atmospheric light head (A) — per-channel estimation
        # -------------------------
        # The head now outputs a 3-element vector (one value per colour
        # channel: A_R, A_G, A_B), enabling wavelength-dependent ambient
        # light modelling.  The output dimension was previously 1 (forcing
        # A_R = A_G = A_B); it is now 3.  For the wavelength-independent
        # case, the three values can be collapsed to a scalar via the
        # `rgb_to_gray_A` helper defined below.
        #
        # Architecture: global average pooling over the bottleneck feature
        # map (B, embed_dim, H/16, W/16) followed by a 1×1 convolution that
        # projects the embed_dim features onto 3 output scalars, and a
        # Sigmoid activation to constrain each estimate to [0, 1].
        self.A_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, embed_dim, 1, 1)
            nn.Conv2d(embed_dim, 3, 1),       # (B, 3, 1, 1) — one scalar per channel
            nn.Sigmoid()                      # A_c in [0, 1],  c ∈ {R, G, B}
        )

        # -------------------------
        # CNN decoder (upsample x16)
        # -------------------------
        # Skip connections from enc3 (x3) and enc2 (x2) and enc1 (x1)
        # improve spatial fidelity of the transmission map.
        self.up1 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2)  # 30x32 -> 60x64
        self.dec1 = nn.Sequential(ConvBNReLU(128 + 128, 128), ConvBNReLU(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        # 60x64 -> 120x128
        self.dec2 = nn.Sequential(ConvBNReLU(64 + 64, 64), ConvBNReLU(64, 64))

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)         # 120x128 -> 240x256
        self.dec3 = nn.Sequential(ConvBNReLU(32 + 32, 32), ConvBNReLU(32, 32))  # +32 from enc1

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)         # 240x256 -> 480x512
        self.dec4 = nn.Sequential(ConvBNReLU(16, 16), ConvBNReLU(16, 16))

        self.head = nn.Conv2d(16, out_ch, kernel_size=1)

        # Initialize pos_embed small (helps stability)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x

        # --- Encoder ---
        x1 = self.enc1(x)          # (B, 32,       H,    W)
        x  = self.down1(x1)        # (B, 32,       H/2,  W/2)

        x2 = self.enc2(x)          # (B, 64,       H/2,  W/2)
        x  = self.down2(x2)        # (B, 64,       H/4,  W/4)

        x3 = self.enc3(x)          # (B, 128,      H/4,  W/4)
        x  = self.down3(x3)        # (B, 128,      H/8,  W/8)

        x4 = self.enc4(x)          # (B, embed_dim, H/8,  W/8)
        x  = self.down4(x4)        # (B, embed_dim, H/16, W/16)

        # ── Atmospheric light estimation (bottleneck, before Transformer) ──
        # A_raw : (B, 3, 1, 1)  →  squeeze to (B, 3)
        # Each element A[:, c] is an independent scalar in [0, 1] for
        # channel c, so R, G, B components are free to take distinct values.
        A = self.A_head(x)                    # (B, 3, 1, 1)
        A = A.squeeze(-1).squeeze(-1)         # (B, 3)

        B, C, H16, W16 = x.shape

        # --- Tokenize for Transformer ---
        tokens = x.flatten(2).transpose(1, 2)   # (B, N, C)

        # Interpolate positional embeddings if spatial size differs from
        # the size assumed at construction time (e.g., variable-resolution
        # inference).
        if tokens.shape[1] != self.pos_embed.shape[1]:
            pe = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, self.H16, self.W16)
            pe = F.interpolate(pe, size=(H16, W16), mode="bilinear", align_corners=False)
            pe = pe.flatten(2).transpose(1, 2)
        else:
            pe = self.pos_embed

        tokens = tokens + pe
        tokens = self.pos_drop(tokens)

        # --- Transformer global context ---
        tokens = self.transformer(tokens)       # (B, N, C)
        tokens = self.norm(tokens)

        # --- Back to feature map ---
        x = tokens.transpose(1, 2).reshape(B, C, H16, W16)  # (B, C, H/16, W/16)

        # --- Decoder with skip connections ---
        x = self.up1(x)                         # (B, 128, H/8, W/8)
        if x.shape[-2:] != x3.shape[-2:]:
            x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)                         # (B, 64, H/4, W/4)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        x = self.up3(x)                         # (B, 32, H/2, W/2)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.up4(x)                         # (B, 16, H, W)
        x = self.dec4(x)

        y = torch.sigmoid(self.head(x))         # (B, 1, H, W) — transmission map T

        # Ensure spatial dimensions exactly match the input (guards against
        # off-by-one rounding in strided convolutions for odd-sized inputs).
        if y.shape[-2:] != x_in.shape[-2:]:
            y = F.interpolate(y, size=x_in.shape[-2:], mode="bilinear", align_corners=False)

        return y, A


# ---------------------------------------------------------------------------
# Helper: collapse per-channel A to a scalar (wavelength-independent case)
# ---------------------------------------------------------------------------
def rgb_to_gray_A(A: torch.Tensor) -> torch.Tensor:
    """
    Convert the per-channel atmospheric light vector to a single scalar
    using the standard ITU-R BT.601 luminance coefficients:

        A_gray = 0.299 * A_R + 0.587 * A_G + 0.114 * A_B

    This is appropriate for applications where the ambient illumination is
    wavelength-independent and a single scalar representation suffices.

    Parameters
    ----------
    A : torch.Tensor, shape (B, 3) or (B, 3, 1, 1)
        Per-channel atmospheric light estimates from the model forward pass.

    Returns
    -------
    A_gray : torch.Tensor, shape (B, 1) or (B, 1, 1, 1)
        Grayscale-equivalent scalar atmospheric light, preserving the
        spatial-singleton dimensions of the input for broadcast compatibility
        with image tensors of shape (B, C, H, W).
    """
    # ITU-R BT.601 luminance weights
    w = A.new_tensor([0.299, 0.587, 0.114])   # (3,)

    if A.dim() == 2:                           # (B, 3)
        A_gray = (A * w).sum(dim=1, keepdim=True)   # (B, 1)
    elif A.dim() == 4:                         # (B, 3, 1, 1)
        A_gray = (A * w[None, :, None, None]).sum(dim=1, keepdim=True)  # (B, 1, 1, 1)
    else:
        raise ValueError(f"Unexpected A shape: {A.shape}. Expected (B,3) or (B,3,1,1).")

    return A_gray
