import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# CNN Patch Extractor (stride=16)
# ----------------------------
class ConvStem(nn.Module):
    def __init__(self, in_ch=1, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, stride=2, padding=1),
        )
        # Output: [B, embed_dim, H/16, W/16]

    def forward(self, x):
        return self.conv(x)


# ----------------------------
# Transformer Block
# ----------------------------
class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ----------------------------
# Decoder
# ----------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.block(x)


# ----------------------------
# TransUNet (Fixed 256×256 Output)
# ----------------------------
class TransUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, embed_dim=256, depth=4, heads=4, mlp_ratio=4):
        """
        TransUNet - PatchSize=16, output 256×256, safe for 8GB GPU
        Params ≈ 6.0M
        """
        super().__init__()
        self.encoder = ConvStem(in_ch, embed_dim)

        num_patches = 16 * 16
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, heads=heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder path (x16 → x256)
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, 2, 2)
        self.dec1 = DecoderBlock(embed_dim // 2, embed_dim // 4)
        self.up2 = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 2, 2)
        self.dec2 = DecoderBlock(embed_dim // 8, embed_dim // 8)
        self.up3 = nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2)
        self.dec3 = DecoderBlock(embed_dim // 16, embed_dim // 16)
        self.up4 = nn.ConvTranspose2d(embed_dim // 16, embed_dim // 32, 2, 2)  # ✅ 추가됨
        self.dec4 = DecoderBlock(embed_dim // 32, embed_dim // 32)              # ✅ 추가됨
        self.outc = nn.Conv2d(embed_dim // 32, out_ch, 1)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.encoder(x)  # [B, C, 16, 16]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 256, C]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), 1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x[:, 1:, :].transpose(1, 2).reshape(B, C, H, W)
        x = self.up1(x); x = self.dec1(x)
        x = self.up2(x); x = self.dec2(x)
        x = self.up3(x); x = self.dec3(x)
        x = self.up4(x); x = self.dec4(x)   # ✅ 새 단계
        return self.outc(x)
