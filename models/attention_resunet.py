import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================
# 🔹 Residual Convolutional Block
# ===========================================================
class ResBlock(nn.Module):
    """Basic Residual Block with two 3×3 convolutions."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # Shortcut to match dimensions if needed
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


# ===========================================================
# 🔹 Attention Gate (Channel and Spatially Aligned)
# ===========================================================
class AttentionBlock(nn.Module):
    """
    Attention Gate for feature refinement between encoder and decoder.
    - F_g: channels of gating (decoder) signal
    - F_l: channels of skip connection (encoder)
    - F_int: intermediate reduced channel size
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Ensure both tensors have the same spatial size
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)

        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ===========================================================
# 🔹 Attention ResUNet Model
# ===========================================================
class AttentionResUNet(nn.Module):
    """
    Attention Residual U-Net:
    Combines residual blocks in encoder/decoder paths with attention gates.
    """
    def __init__(self, in_ch=1, out_ch=1, base=35):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = ResBlock(in_ch, base)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base, base * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base * 2, base * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base * 4, base * 8))
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), ResBlock(base * 8, base * 16))

        # ---------- Attention Gates (corrected channel mappings) ----------
        self.att1 = AttentionBlock(F_g=base * 16, F_l=base * 8, F_int=base * 4)
        self.att2 = AttentionBlock(F_g=base * 8, F_l=base * 4, F_int=base * 2)
        self.att3 = AttentionBlock(F_g=base * 4, F_l=base * 2, F_int=base)
        self.att4 = AttentionBlock(F_g=base * 2, F_l=base, F_int=base // 2)

        # ---------- Decoder ----------
        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base * 16, base * 8)

        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base * 8, base * 4)

        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec3 = ResBlock(base * 4, base * 2)

        self.up4 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec4 = ResBlock(base * 2, base)

        # ---------- Output ----------
        self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)

        # ----- Attention-based Skip Connections -----
        x4 = self.att1(x5, x4)
        x3 = self.att2(x4, x3)
        x2 = self.att3(x3, x2)
        x1 = self.att4(x2, x1)

        # ----- Decoder -----
        d1 = self.up1(x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.dec4(d4)

        return self.outc(d4)
