"""
Network Building Blocks for BLUE-Net

This module contains all the sub-network architectures used in BLUE-Net:
1. Basic convolution and residual blocks
2. Channel Attention Block (CAB) and Channel Attention Layer
3. Supervised Attention Module (SAM) and Merge Block for feature fusion
4. U-Net Encoder-Decoder with Cross-Stage Feature Fusion (CSFF)
5. Residual Dense Network (RDN) for transmission map denoising

Author: Thuy Thi Pham
"""

from pdb import set_trace as stx

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# BASIC BUILDING BLOCKS
# ==============================================================================


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    """
    Standard 2D convolution with automatic padding to preserve spatial dimensions.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        bias: Whether to include bias term
        stride: Convolution stride

    Returns:
        nn.Conv2d layer
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def conv_down(in_chn, out_chn, bias=False):
    """
    Downsampling convolution (reduces spatial dimensions by 2x).

    Uses kernel_size=4, stride=2, padding=1 to halve H and W.
    """
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    """
    Default convolution layer with padding.
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        stride=stride,
        bias=bias,
    )


class ResBlock(nn.Module):
    """
    Residual block with optional batch normalization.

    Structure: Conv -> [BN] -> Act -> Conv -> [BN] -> Add residual
    """

    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.PReLU(),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# ==============================================================================
# CHANNEL ATTENTION MECHANISMS
# ==============================================================================


class CAB(nn.Module):
    """
    Channel Attention Block: Conv -> Act -> Conv -> Channel Attention -> Residual

    Enhances informative channels and suppresses less useful ones through
    global average pooling and channel-wise weights.
    """

    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# ==============================================================================
# SUPERVISED ATTENTION MODULE (SAM)
# ==============================================================================


class SAM(nn.Module):
    """
    Supervised Attention Module for progressive image refinement.

    Combines feature-level attention with image-level residual learning.
    Outputs both updated features and refined image for multi-scale supervision.
    """

    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img


class mergeblock(nn.Module):
    """
    Subspace Projection Merge Block for cross-stage feature fusion.

    Projects features from previous stage onto a learned subspace before merging
    with current features. This reduces redundancy and focuses on complementary information.

    Steps:
    1. Learn subspace basis V from concatenated features
    2. Project previous stage features (bridge) onto V
    3. Reconstruct projected features
    4. Merge with current features via concatenation + conv
    """

    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim  # Dimensionality of learned subspace
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        """
        Args:
            x: Current stage features [B, C, H, W]
            bridge: Previous stage features [B, C, H, W]

        Returns:
            Merged features with residual connection [B, C, H, W]
        """
        out = torch.cat([x, bridge], 1)  # [B, 2C, H, W]
        b_, c_, h_, w_ = bridge.shape

        # Learn subspace basis V: [B, subspace_dim, H*W]
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))  # Normalize
        V = V_t.permute(0, 2, 1)  # [B, H*W, subspace_dim]

        # Compute projection matrix: P = V(V^T V)^-1 V^T
        mat = torch.matmul(V_t, V)  # [B, subspace_dim, subspace_dim]
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)  # [B, subspace_dim, H*W]

        # Project bridge features onto subspace
        bridge_ = bridge.view(b_, c_, h_ * w_)
        project_feature = torch.matmul(
            project_mat, bridge_.permute(0, 2, 1)
        )  # [B, subspace_dim, C]
        bridge = (
            torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        )  # Reconstruct

        # Merge and add residual
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out + x  # Residual connection


# ==============================================================================
# U-NET ENCODER-DECODER WITH CROSS-STAGE FEATURE FUSION (CSFF)
# ==============================================================================


class Encoder(nn.Module):
    """
    Multi-scale U-Net Encoder with Cross-Stage Feature Fusion (CSFF).

    Progressive downsampling with skip connections. Supports CSFF by incorporating
    encoder/decoder features from previous IPMM stage.
    """

    def __init__(
        self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5
    ):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()  # []
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(
                    in_size=n_feat + scale_unetfeats * i,
                    out_size=n_feat + scale_unetfeats * (i + 1),
                    downsample=True,
                    relu_slope=0.2,
                    use_csff=csff,
                    use_HIN=True,
                )
            )
        self.body.append(
            UNetConvBlock(
                in_size=n_feat + scale_unetfeats * (depth - 1),
                out_size=n_feat + scale_unetfeats * (depth - 1),
                downsample=False,
                relu_slope=0.2,
                use_csff=csff,
                use_HIN=True,
            )
        )

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetConvBlock(nn.Module):
    def __init__(
        self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False
    ):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(
                self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True
            )
            out = out * F.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = UNetConvBlock(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=5
    ):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()  # []
        for i in range(depth - 1):
            self.body.append(
                UNetUpBlock(
                    in_size=n_feat + scale_unetfeats * (depth - i - 1),
                    out_size=n_feat + scale_unetfeats * (depth - i - 2),
                    relu_slope=0.2,
                )
            )
            self.skip_conv.append(
                nn.Conv2d(
                    n_feat + scale_unetfeats * (depth - i - 1),
                    n_feat + scale_unetfeats * (depth - i - 2),
                    3,
                    1,
                    1,
                )
            )

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i - 1]))
            res.append(x)

        return res


# ==============================================================================
# RESIZING MODULES
# ==============================================================================


class DownSample(nn.Module):
    """
    Downsample feature maps by 2x using bilinear interpolation + 1x1 conv.
    """

    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels, in_channels + s_factor, 1, stride=1, padding=0, bias=False
            ),
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.up(x)
        return x


# ==============================================================================
# STANDARD U-NET BLOCKS (Not used in main BLUE-Net, included for completeness)
# ==============================================================================


class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv -> BN -> ReLU) * 2

    Standard U-Net building block with batch normalization.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


# ==============================================================================
# RESIDUAL DENSE NETWORK (RDN) - Used as Z-Net for Transmission Denoising
# ==============================================================================


class DenseLayer(nn.Module):
    """
    Single dense layer: Conv -> ReLU -> Concatenate with input.

    Dense connections allow each layer to access features from all previous layers.
    """

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Concatenate input with conv output (dense connection)."""
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    """
    Residual Dense Block (RDB): stack of dense layers + local feature fusion + residual.

    Structure:
    1. Multiple dense layers with concatenation
    2. Local Feature Fusion (LFF): 1x1 conv to compress concatenated features
    3. Local Residual Learning: add input to LFF output
    """

    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        # Stack of dense layers
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate)
                for i in range(num_layers)
            ]
        )

        # Local feature fusion: compress concatenated features back to growth_rate channels
        self.lff = nn.Conv2d(
            in_channels + growth_rate * num_layers, growth_rate, kernel_size=1
        )

    def forward(self, x):
        """Local residual learning: output = input + LFF(dense_layers(input))."""
        return x + self.lff(self.layers(x))


class RDN(nn.Module):
    """
    Residual Dense Network for transmission map denoising.

    Architecture:
    1. Shallow Feature Extraction (SFE): 2 conv layers
    2. Multiple Residual Dense Blocks (RDBs) with dense connections
    3. Global Feature Fusion (GFF): aggregate features from all RDBs
    4. Output layer: final 3-channel transmission map

    Used as Z-Net in BasicBlock to denoise transmission estimates.
    """

    def __init__(
        self,
        input_channel,
        output_channel=3,
        num_features=64,
        growth_rate=64,
        num_blocks=2,
        num_layers=1,
    ):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(
            input_channel, num_features, kernel_size=3, padding=3 // 2
        )
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )

        # up-sampling
        # assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.output = nn.Conv2d(self.G0, output_channel, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        input = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        output = self.output(x)
        # output = output + input
        return output
