import math
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> ReLU -> Conv2d -> ReLU) block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U-Net with skip connections via concatenation.
    Architecture: Encoder (DoubleConv + MaxPool) -> Bottleneck -> Decoder (ConvTranspose2d + Cat + DoubleConv).
    Global residual: Output = Input + UNet(Input).
    """

    def __init__(self, base_ch: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_c = 1
        out_c = base_ch
        for _ in range(depth):
            self.enc_blocks.append(DoubleConv(in_c, out_c))
            self.pools.append(nn.MaxPool2d(2))
            in_c = out_c
            out_c *= 2

        self.bottleneck = DoubleConv(in_c, out_c)

        in_c = out_c
        out_c = in_c // 2
        for _ in range(depth):
            self.up_blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=False))
            # Concatenation doubles the input channels to the decoder DoubleConv
            self.dec_blocks.append(DoubleConv(out_c * 2, out_c))
            in_c = out_c
            out_c = in_c // 2

        self.out_conv = nn.Conv2d(in_c, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        divisor = 2 ** self.depth
        assert h % divisor == 0 and w % divisor == 0, f"H,W must be divisible by {divisor}"

        skips = []
        curr = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            curr = enc(curr)
            skips.append(curr)
            curr = pool(curr)

        curr = self.bottleneck(curr)

        for i, (up, dec) in enumerate(zip(self.up_blocks, self.dec_blocks)):
            curr = up(curr)
            skip = skips[-(i + 1)]
            curr = torch.cat([curr, skip], dim=1)
            curr = dec(curr)

        return x + self.out_conv(curr)


class HourglassNoSkip(nn.Module):
    """
    Hourglass U-Net (formerly ResidualUNet). Identical topology to UNet but STRICTLY NO skip connections.
    Global residual: Output = Input + Net(Input).
    """

    def __init__(self, base_ch: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_c = 1
        out_c = base_ch
        for _ in range(depth):
            self.enc_blocks.append(DoubleConv(in_c, out_c))
            self.pools.append(nn.MaxPool2d(2))
            in_c = out_c
            out_c *= 2

        self.bottleneck = DoubleConv(in_c, out_c)

        in_c = out_c
        out_c = in_c // 2
        for _ in range(depth):
            self.up_blocks.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=False))
            # No concatenation, so DoubleConv takes only the upsampled channels
            self.dec_blocks.append(DoubleConv(out_c, out_c))
            in_c = out_c
            out_c = in_c // 2

        self.out_conv = nn.Conv2d(in_c, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        divisor = 2 ** self.depth
        assert h % divisor == 0 and w % divisor == 0, f"H,W must be divisible by {divisor}"

        curr = x
        for enc, pool in zip(self.enc_blocks, self.pools):
            curr = enc(curr)
            curr = pool(curr)

        curr = self.bottleneck(curr)

        for up, dec in zip(self.up_blocks, self.dec_blocks):
            curr = up(curr)
            curr = dec(curr)

        return x + self.out_conv(curr)


class PlainCNN(nn.Module):
    """
    Plain residual CNN (no downsampling) mapping 1->width->width...->1.
    Output = Input + Net(Input).
    """

    def __init__(self, depth: int = 8, width: int = 64) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"Depth must be >= 2, got {depth}")
            
        layers = []
        layers.append(nn.Conv2d(1, width, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv2d(width, 1, kernel_size=3, padding=1, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def count_params(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_plaincnn(target_params: int, depth: int = 8) -> PlainCNN:
    """
    Analytically solves for the channel width of a PlainCNN of a given depth
    to approximate target_params within a 5% error margin.
    """
    # PlainCNN parameter equation (no bias):
    # Total = (1 * width * 9) + (depth - 2) * (width * width * 9) + (width * 1 * 9)
    # Total = 9 * (depth - 2) * width^2 + 18 * width
    # Set to zero for quadratic formula: 9*(depth-2)*width^2 + 18*width - target_params = 0
    A = 9 * (depth - 2)
    B = 18
    C = -target_params
    
    if A > 0:
        width = int(round((-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)))
    else:
        width = int(round(target_params / B))
        
    return PlainCNN(depth=depth, width=width)


# =============================================================================
# Unit, Shape & Parameter Alignment Tests
# =============================================================================

def test_all_output_shapes() -> None:
    x = torch.randn(2, 1, 128, 128)
    for model in [UNet(base_ch=16, depth=4), HourglassNoSkip(base_ch=16, depth=4), PlainCNN(depth=8, width=64)]:
        out = model(x)
        assert out.shape == (2, 1, 128, 128), f"Expected (2, 1, 128, 128) for {model.__class__.__name__}, got {out.shape}"


def test_unet_params() -> None:
    unet = UNet(base_ch=16, depth=4)
    params = count_params(unet)
    assert 1_500_000 <= params <= 2_500_000, f"Expected 1.5M - 2.5M params for UNet, got {params}"


def test_hourglass_params() -> None:
    unet = UNet(base_ch=16, depth=4)
    hg = HourglassNoSkip(base_ch=16, depth=4)
    assert count_params(hg) < count_params(unet), "HourglassNoSkip must have fewer parameters than UNet"


def test_match_plaincnn() -> None:
    unet = UNet(base_ch=16, depth=4)
    target = count_params(unet)
    matched_cnn = match_plaincnn(target, depth=8)
    matched_params = count_params(matched_cnn)
    
    diff_ratio = abs(matched_params - target) / target
    assert diff_ratio <= 0.05, f"Expected <5% difference, got {diff_ratio*100:.2f}% (Target: {target}, Matched: {matched_params})"


def test_unet_wiring() -> None:
    """Proves the first decoder DoubleConv is correctly dimensioned to take concatenated skip features."""
    unet = UNet(base_ch=16, depth=4)
    first_up_out_ch = unet.up_blocks[0].out_channels
    first_dec_in_ch = unet.dec_blocks[0].block[0].in_channels
    
    assert first_dec_in_ch == 2 * first_up_out_ch, (
        f"Decoder input channels ({first_dec_in_ch}) must equal exactly 2x upsampled channels "
        f"({first_up_out_ch}) to accommodate the symmetric encoder skip connection."
    )


if __name__ == "__main__":
    test_all_output_shapes()
    test_unet_params()
    test_hourglass_params()
    test_match_plaincnn()
    test_unet_wiring()
    print("All models.py tests passed.")