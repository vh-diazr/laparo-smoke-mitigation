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
        -> 1-channel output map

    Input : (N, 3, H, W)  e.g. (N, 3, 480, 512)
    Output: (N, 1, H, W)
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
        # self.down1 = nn.MaxPool2d(2)
        self.down1 = nn.Conv2d(32, 32, kernel_size=2, stride=2, bias=False)
    

        self.enc2 = nn.Sequential(ConvBNReLU(32, 64), ConvBNReLU(64, 64))
        # self.down2 = nn.MaxPool2d(2)
        self.down2 = nn.Conv2d(64, 64, kernel_size=2, stride=2, bias=False)

        self.enc3 = nn.Sequential(ConvBNReLU(64, 128), ConvBNReLU(128, 128))
        # self.down3 = nn.MaxPool2d(2)
        self.down3 = nn.Conv2d(128,       128,       kernel_size=2, stride=2, bias=False)

        self.enc4 = nn.Sequential(ConvBNReLU(128, embed_dim), ConvBNReLU(embed_dim, embed_dim))
        # self.down4 = nn.MaxPool2d(2)
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

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        # -------------------------
        # Atmospheric light head (A)          
        # -------------------------
        self.A_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, embed_dim, 1, 1)
            nn.Conv2d(embed_dim, 1, 1),       # (B, 1, 1, 1) single scalar
            nn.Sigmoid()                      # A in [0, 1]
        )

        # -------------------------
        # CNN decoder (upsample x16)
        # -------------------------
        # Optional skip connections (simple concatenation + conv) improve detail.
        # We include skips from enc3 and enc2 for better spatial fidelity.
        self.up1 = nn.ConvTranspose2d(embed_dim, 128, kernel_size=2, stride=2)  # 30x32 -> 60x64
        self.dec1 = nn.Sequential(ConvBNReLU(128 + 128, 128), ConvBNReLU(128, 128))

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)        # 60x64 -> 120x128
        self.dec2 = nn.Sequential(ConvBNReLU(64 + 64, 64), ConvBNReLU(64, 64))

        # self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)         # 120x128 -> 240x256
        # self.dec3 = nn.Sequential(ConvBNReLU(32, 32), ConvBNReLU(32, 32))
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ConvBNReLU(32 + 32, 32), ConvBNReLU(32, 32))  # +32 from enc1

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)         # 240x256 -> 480x512
        self.dec4 = nn.Sequential(ConvBNReLU(16, 16), ConvBNReLU(16, 16))

        self.head = nn.Conv2d(16, out_ch, kernel_size=1)

        # Initialize pos_embed small (helps stability)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        # --- Encoder ---
        x1 = self.enc1(x)          # (B,32,H,W)
        x  = self.down1(x1)        # (B,32,H/2,W/2)

        x2 = self.enc2(x)          # (B,64,H/2,W/2)
        x  = self.down2(x2)        # (B,64,H/4,W/4)

        x3 = self.enc3(x)          # (B,128,H/4,W/4)
        x  = self.down3(x3)        # (B,128,H/8,W/8)

        x4 = self.enc4(x)          # (B,embed,H/8,W/8)
        x  = self.down4(x4)        # (B,embed,H/16,W/16) -> (B,embed,30,32) for 480x512

        # ── A prediction (from bottleneck, before Transformer) ──
        A = self.A_head(x)         # (B, 3, 1, 1)
        A = A.squeeze(-1).squeeze(-1)   # (B, 3)
        A = A.expand(-1, 3)                    # (B, 3) — R=G=B guaranteed

        B, C, H16, W16 = x.shape
        # Safety: if you ever change input size, you can interpolate pos embeddings,
        # but for now we assume fixed 480x512.
        # assert H16 * W16 == self.num_tokens, "Token grid size mismatch. Check img_hw or input size."

        # # --- Tokenize for Transformer ---
        # tokens = x.flatten(2).transpose(1, 2)   # (B, N, C)
        # tokens = tokens + self.pos_embed
        # tokens = self.pos_drop(tokens)
        # --- Tokenize for Transformer ---
        tokens = x.flatten(2).transpose(1, 2)   # (B, N, C)

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
        x = tokens.transpose(1, 2).reshape(B, C, H16, W16)  # (B,C,H/16,W/16)

        # --- Decoder with skips ---
        x = self.up1(x)                         # (B,128,H/8,W/8)
        if x.shape[-2:] != x3.shape[-2:]:
            x = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.up2(x)                         # (B,64,H/4,W/4)
        if x.shape[-2:] != x2.shape[-2:]:
            x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        # x = self.up3(x)                         # (B,32,H/2,W/2)
        # x = self.dec3(x)
        x = self.up3(x)
        if x.shape[-2:] != x1.shape[-2:]:
            x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        x = self.up4(x)                         # (B,16,H,W)
        x = self.dec4(x)

        #y = self.head(x)                        # (B,1,H,W)
        y = torch.sigmoid(self.head(x))

        # Ensure output matches input size exactly
        if y.shape[-2:] != x_in.shape[-2:]:
            y = F.interpolate(y, size=x_in.shape[-2:], mode="bilinear", align_corners=False)

        return y, A
