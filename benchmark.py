import math
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

BLUENET_PATH = "./models/bluenet.pth"
SSUIE_PATH = "./models/ss-uie.pth"

DATASETS = {
    "LSUI": {"input": "./dataset/LSUI/input", "gt": "./dataset/LSUI/GT"},
    "UIEB": {"input": "./dataset/UIEB/raw-890", "gt": "./dataset/UIEB/reference-890"},
}

SAVE_DIR = "outputs"
CSV_NAME = "results.csv"

IMG_SIZE = (256, 256)
BLUENET_LAYERS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
DEVICE = device

print(f"Using device: {device}")


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        stride=stride,
        bias=bias,
    )


class ResBlock(nn.Module):
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


class CAB(nn.Module):
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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


class SAM(nn.Module):
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
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        with torch.amp.autocast("cuda", enabled=False):
            mat = torch.matmul(V_t.float(), V.float())
            mat_inv = torch.inverse(mat)
            project_mat = torch.matmul(mat_inv, V_t.float())
            bridge_ = bridge.view(b_, c_, h_ * w_)
            project_feature = torch.matmul(
                project_mat, bridge_.permute(0, 2, 1).float()
            )
            bridge = (
                torch.matmul(V.float(), project_feature)
                .permute(0, 2, 1)
                .view(b_, c_, h_, w_)
            )
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out + x


class Encoder(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff, depth=5
    ):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()
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
        self.skip_conv = nn.ModuleList()
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


class DownSample(nn.Module):
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


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
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


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[
                DenseLayer(in_channels + growth_rate * i, growth_rate)
                for i in range(num_layers)
            ]
        )
        self.lff = nn.Conv2d(
            in_channels + growth_rate * num_layers, growth_rate, kernel_size=1
        )

    def forward(self, x):
        return x + self.lff(self.layers(x))


class RDN(nn.Module):
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
        self.sfe1 = nn.Conv2d(
            input_channel, num_features, kernel_size=3, padding=3 // 2
        )
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )
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
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        output = self.output(x)
        return output


def get_mean_value(batch):
    batch_size = batch.shape[0]
    list_mean_sorted = []
    list_indices = []
    largest_index = []
    medium_index = []
    smallest_index = []
    largest_channel = []
    medium_channel = []
    smallest_channel = []
    for bs in range(batch_size):
        image = batch[bs, :, :, :]
        mean = torch.mean(image, (2, 1))
        mean_I_sorted, indices = torch.sort(mean)
        list_mean_sorted.append(mean_I_sorted)
        list_indices.append(indices)
        largest_index.append(indices[2])
        medium_index.append(indices[1])
        smallest_index.append(indices[0])
        largest_channel.append(torch.unsqueeze(image[indices[2], :, :], 0))
        medium_channel.append(torch.unsqueeze(image[indices[1], :, :], 0))
        smallest_channel.append(torch.unsqueeze(image[indices[0], :, :], 0))
    list_mean_sorted = torch.stack(list_mean_sorted)
    list_indices = torch.stack(list_indices)
    largest_index = torch.stack(largest_index)
    medium_index = torch.stack(medium_index)
    smallest_index = torch.stack(smallest_index)
    largest_channel = torch.stack(largest_channel)
    medium_channel = torch.stack(medium_channel)
    smallest_channel = torch.stack(smallest_channel)
    return (
        list_mean_sorted,
        list_indices,
        largest_channel,
        medium_channel,
        smallest_channel,
        largest_index,
        medium_index,
        smallest_index,
    )


def mapping_index(batch, value, index):
    batch_size = batch.shape[0]
    new_batch = []
    for bs in range(batch_size):
        image = batch[bs, :, :, :]
        image[index[bs], :, :] = value[bs]
        new_batch.append(image)
    new_batch = torch.stack(new_batch)
    return new_batch


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        Z_Net = [RDN(3)]
        self.Z_Net = nn.Sequential(*Z_Net)
        self.t_1D_Net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.alpha = nn.Parameter(torch.tensor([3.001]), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor([3.001]), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor([3.001]), requires_grad=True)
        self.lambda_1 = nn.Parameter(torch.tensor([1.001]), requires_grad=True)
        self.lambda_2 = nn.Parameter(torch.tensor([1.001]), requires_grad=True)

    def forward(self, I, t_p, B_p, B, t, J, Y, Z, Q, R, u, v, w_1, w_2, eps=1e-6):
        alpha = self.alpha
        beta = self.beta
        eta = self.eta
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        gamma_1 = 0.3
        gamma_2 = 0.7
        (
            list_mean_sorted,
            list_indices,
            J_l,
            J_m,
            J_s,
            largest_index,
            medium_index,
            smallest_index,
        ) = get_mean_value(J)
        J_l_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 2], 1), 1), 1
        ).to(DEVICE)
        J_m_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 1], 1), 1), 1
        ).to(DEVICE)
        J_s_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 0], 1), 1), 1
        ).to(DEVICE)
        J_m_bar = J_l - u + (1.0 / lambda_1) * w_1
        J_s_bar = J_l - v + (1.0 / lambda_2) * w_2
        J_l = J_l.to(DEVICE)
        J_m = J_m.to(DEVICE)
        J_s = J_s.to(DEVICE)
        J_m = J_m + torch.mul(J_l_bar - J_m_bar, J_l)
        J_s = J_s + torch.mul(J_l_bar - J_s_bar, J_l)
        J = mapping_index(J.clone(), J_m.clone(), medium_index)
        J = mapping_index(J.clone(), J_s.clone(), smallest_index)
        u = torch.sign((J_l_bar - J_m_bar + (1.0 / lambda_1) * w_1)) * F.relu(
            torch.abs((J_l_bar - J_m_bar + (1.0 / lambda_1) * w_1)) - (1.0 / lambda_1),
            inplace=False,
        )
        v = torch.sign((J_l_bar - J_s_bar + (1.0 / lambda_2) * w_2)) * F.relu(
            torch.abs((J_l_bar - J_s_bar + (1.0 / lambda_2) * w_2)) - (1.0 / lambda_2),
            inplace=False,
        )
        w_1 = w_1 + lambda_1 * (J_l_bar - J_m_bar - u)
        w_2 = w_2 + lambda_2 * (J_l_bar - J_s_bar - v)
        D = torch.ones(I.shape).to(DEVICE)
        B = (gamma_1 * B_p - (J * t - I) * (1 - t)) / ((1.0 - t) * (1 - t) + gamma_1)
        B = torch.mean(B, (2, 3), True)
        B = B * D
        t = (gamma_2 * t_p + eta * Z - R - (B - I) * (J - B)) / (
            (J - B) * (J - B) + gamma_2 + eta
        )
        t = self.t_1D_Net(t)
        t = torch.cat((t, t, t), 1)
        J = (beta * Y - Q - (B * (1.0 - t) - I) * t) / (t * t + beta)
        Z = self.Z_Net(t + (1.0 / eta) * R)
        Q = Q + beta * (J - Y)
        R = R + eta * (t - Z)
        return B, t, J, Y, Z, Q, R, u, v, w_1, w_2, beta


class IPMM(nn.Module):
    def __init__(
        self,
        in_c=3,
        out_c=3,
        n_feat=80,
        scale_unetfeats=48,
        scale_orsnetfeats=32,
        num_cab=8,
        kernel_size=3,
        reduction=4,
        bias=False,
    ):
        super(IPMM, self).__init__()
        act = nn.PReLU()
        self.shallow_feat2 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )
        self.stage2_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            depth=4,
            csff=True,
        )
        self.stage2_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4
        )
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, 3, True)

    def forward(self, x2_img, stage1_img, feat1, res1, x2_samfeats):
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        res2 = self.stage2_decoder(feat_fin2, feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
        return x3_samfeats, stage2_img, feat2, res2


class BLUE_Net(torch.nn.Module):
    def __init__(self, LayerNo):
        super(BLUE_Net, self).__init__()
        self.LayerNo = LayerNo
        net_layers = []
        for i in range(LayerNo):
            net_layers.append(BasicBlock())
        self.uunet = nn.ModuleList(net_layers)
        in_c = 3
        out_c = 3
        n_feat = 40
        scale_unetfeats = 20
        scale_orsnetfeats = 16
        num_cab = 8
        kernel_size = 3
        reduction = 4
        bias = False
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(
            conv(3, n_feat, kernel_size, bias=bias),
            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
        )
        self.stage1_encoder = Encoder(
            n_feat,
            kernel_size,
            reduction,
            act,
            bias,
            scale_unetfeats,
            depth=4,
            csff=True,
        )
        self.stage1_decoder = Decoder(
            n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4
        )
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, 3, True)
        self.basic = IPMM(
            in_c=3,
            out_c=3,
            n_feat=40,
            scale_unetfeats=20,
            scale_orsnetfeats=16,
            num_cab=8,
            kernel_size=3,
            reduction=4,
            bias=False,
        )

    def forward(self, I, t_p, B_p):
        bs, _, _, _ = I.shape
        B = torch.zeros((bs, 3, 1, 1)).to(DEVICE)
        t = torch.zeros(I.shape).to(DEVICE)
        J = I.to(DEVICE)
        X = torch.zeros(I.shape).to(DEVICE)
        Y = torch.zeros(I.shape).to(DEVICE)
        Z = torch.zeros(I.shape).to(DEVICE)
        P = torch.zeros(I.shape).to(DEVICE)
        Q = torch.zeros(I.shape).to(DEVICE)
        R = torch.zeros(I.shape).to(DEVICE)
        u = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        v = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        w_1 = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        w_2 = torch.zeros((bs, 1, 1, 1)).to(DEVICE)
        list_output = []
        list_B = []
        list_t = []
        beta = torch.tensor([3.001]).to(DEVICE)
        x1_img = J + (1.0 / beta) * Q
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)
        Y = stage1_img
        for j in range(self.LayerNo):
            [B, t, J, Y, Z, Q, R, u, v, w_1, w_2, beta] = self.uunet[j](
                I, t_p, B_p, B, t, J, Y, Z, Q, R, u, v, w_1, w_2
            )
            img = J + (1.0 / beta) * Q
            x2_samfeats, stage1_img, feat1, res1 = self.basic(
                img, stage1_img, feat1, res1, x2_samfeats
            )
            Y = stage1_img
            if j < self.LayerNo - 1:
                list_output.append(J.detach())
                list_B.append(B.detach())
                list_t.append(t.detach())
            else:
                list_output.append(J)
                list_B.append(B)
                list_t.append(t)
        return list_output, list_B, list_t


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0, **kwargs):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            d_conv,
            groups=self.d_inner,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        self.x_proj_weight = nn.Parameter(
            torch.empty(4, (self.dt_rank + d_state * 2), self.d_inner)
        )
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self.A_logs = nn.Parameter(
            torch.log(
                repeat(
                    torch.arange(1, d_state + 1, dtype=torch.float32),
                    "n -> d n",
                    d=self.d_inner,
                )
            )
            .flatten(0, 1)
            .repeat(4, 1)
        )
        self.Ds = nn.Parameter(torch.ones(self.d_inner * 4))
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.xavier_uniform_(self.x_proj_weight)
        nn.init.xavier_uniform_(self.dt_projs_weight)
        nn.init.zeros_(self.dt_projs_bias)

    def forward(self, x):
        B, H, W, C = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        L = H * W
        xh = torch.stack(
            [
                x.reshape(B, -1, L),
                torch.transpose(x, 2, 3).contiguous().reshape(B, -1, L),
            ],
            1,
        ).reshape(B, 2, -1, L)
        xs = torch.cat([xh, torch.flip(xh, [-1])], 1)
        if self.selective_scan:
            xd = torch.einsum(
                "bkdl,kcd->bkcl", xs.reshape(B, 4, -1, L), self.x_proj_weight
            )
            dts, Bs, Cs = torch.split(xd, [self.dt_rank, 16, 16], 2)
            dts = torch.einsum(
                "bkrl,kdr->bkdl", dts.reshape(B, 4, -1, L), self.dt_projs_weight
            )
            oy = self.selective_scan(
                xs.float().reshape(B, -1, L),
                dts.float().reshape(B, -1, L),
                -torch.exp(self.A_logs.float()).reshape(-1, 16),
                Bs.float().reshape(B, 4, -1, L),
                Cs.float().reshape(B, 4, -1, L),
                self.Ds.float().reshape(-1),
                z=None,
                delta_bias=self.dt_projs_bias.float().reshape(-1),
                delta_softplus=True,
            ).reshape(B, 4, -1, L)
            y = (
                oy[:, 0]
                + torch.flip(oy[:, 2], [-1])
                + torch.transpose(
                    (oy[:, 1] + torch.flip(oy[:, 3], [-1])).reshape(B, -1, W, H), 2, 3
                ).reshape(B, -1, L)
            )
        else:
            y = xs.sum(1).reshape(B, -1, L)
        y = self.out_norm(torch.transpose(y, 1, 2).reshape(B, H, W, -1))
        y = y * F.silu(z)
        return self.out_proj(y)


class SF_Block(nn.Module):
    def __init__(self, c, H, W):
        super().__init__()
        self.sp = nn.Parameter(torch.randn(H, W // 2 + 1, c, 2) * 0.02)
        self.ma = SS2D(c)
        self.norm = nn.LayerNorm(c)
        self.fuse = nn.Conv2d(c * 2, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x.float(), norm="ortho")
        xf = xf * torch.view_as_complex(self.sp).permute(2, 0, 1).unsqueeze(0)
        xs = torch.fft.irfft2(xf, s=(H, W), norm="ortho")
        xm = self.ma(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x + self.fuse(torch.cat([xs + x, xm + x], 1))


class MemoryBlock(nn.Module):
    def __init__(self, c, nr, nm, H, W):
        super().__init__()
        self.blocks = nn.ModuleList([SF_Block(c, H, W) for _ in range(nr)])
        self.gate = nn.Conv2d((nr + nm) * c, c, 1)

    def forward(self, x, ys):
        xs = []
        for b in self.blocks:
            x = b(x) + x
            xs.append(x)
        g = self.gate(torch.cat(xs + ys, 1))
        ys.append(g)
        return g


class SS_UIE_model(nn.Module):
    def __init__(self, c=16, nm=4, nr=4, H=256, W=256):
        super().__init__()
        self.e1 = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, c, 3, 1, 1))
        self.e2 = nn.Sequential(
            nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c * 2, 3, 1, 1)
        )
        self.e3 = nn.Sequential(
            nn.BatchNorm2d(c * 2), nn.ReLU(), nn.Conv2d(c * 2, c * 4, 3, 1, 1)
        )
        self.p = nn.MaxPool2d(2, 2)
        self.f = nn.Conv2d(c * 4, c * 4, 3, 1, 1)
        self.m = nn.ModuleList(
            [MemoryBlock(c * 4, nr, i + 1, H // 4, W // 4) for i in range(nm)]
        )
        self.r1 = nn.Conv2d(c * 4, c * 2, 3, 1, 1)
        self.r2 = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.r3 = nn.Conv2d(c, 3, 3, 1, 1)
        self.w = nn.Parameter(torch.ones(1, nm) / nm)

    def forward(self, x):
        r0 = x
        x = self.e1(x)
        r1 = x
        x = self.p(self.e2(x))
        r2 = x
        x = self.p(self.e3(x))
        r3 = x
        ys = [x]
        feats = []
        for b in self.m:
            x = b(x, ys)
            feats.append(x)
        p = (
            sum([(self.f(feats[i]) + r3) * self.w[0][i] for i in range(len(feats))])
            / self.w.sum()
        )
        p = self.r1(F.interpolate(p, scale_factor=2)) + r2
        p = self.r2(F.interpolate(p, scale_factor=2)) + r1
        return self.r3(p) + r0


class GuidedFilter:
    def __init__(self, I, radius, epsilon):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        if img.dtype == np.float32:
            return img
        return np.float32(img) / 255.0

    def _initFilter(self):
        I = self._I
        r = self._radius
        eps = self._epsilon

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir**2, (r, r)) - self._Ir_mean**2 + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean
        Igg_var = cv2.blur(Ig**2, (r, r)) - self._Ig_mean**2 + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean
        Ibb_var = cv2.blur(Ib**2, (r, r)) - self._Ib_mean**2 + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var
        self._Irr_inv = Irr_inv / cov
        self._Irg_inv = Irg_inv / cov
        self._Irb_inv = Irb_inv / cov
        self._Igg_inv = Igg_inv / cov
        self._Igb_inv = Igb_inv / cov
        self._Ibb_inv = Ibb_inv / cov

    def filter(self, p):
        r = self._radius
        I = self._I
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        p = self._toFloatImg(p)
        p_mean = cv2.blur(p, (r, r))
        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean

        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean

        ar = cv2.blur(ar, (r, r))
        ag = cv2.blur(ag, (r, r))
        ab = cv2.blur(ab, (r, r))
        b = cv2.blur(b, (r, r))

        return ar * Ir + ag * Ig + ab * Ib + b


def get_attenuation(image, gamma=1.2):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    vals = [np.mean(1 - b**gamma), np.mean(1 - g**gamma), np.mean(1 - r**gamma)]
    return np.argsort(vals)


def getMaxChannel(img, blockSize):
    kernel = np.ones((blockSize, blockSize), np.uint8)
    return cv2.dilate(img, kernel)


def DepthMap(img, blockSize, idx):
    c_star = img[:, :, idx[-1]]
    c = np.maximum(img[:, :, idx[0]], img[:, :, idx[1]])
    max1 = getMaxChannel(c_star, blockSize)
    max2 = getMaxChannel(c, blockSize)
    return max1 - max2


def estimateBackgroundLight(image):
    idx = get_attenuation(image)
    depth = DepthMap(image, 9, idx)
    pos = np.unravel_index(np.argmin(depth), depth.shape)
    return image[pos[0], pos[1], :]


def generate_priors(dataset_name, info):
    input_dir = info["input"]
    base_dir = str(Path(input_dir).parent)

    t_dir = os.path.join(base_dir, "t_prior")
    b_dir = os.path.join(base_dir, "b_prior")
    os.makedirs(t_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir))

    print(f"\nGenerating priors for {dataset_name}...")

    for fname in tqdm(files):
        stem = Path(fname).stem
        t_path = os.path.join(t_dir, stem + ".png")
        b_path = os.path.join(b_dir, stem + ".png")

        if os.path.exists(t_path) and os.path.exists(b_path):
            continue

        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        image = img / 255.0

        idx = get_attenuation(image)
        depth = DepthMap(image, 9, idx)
        t = depth + (1 - np.max(depth))
        t = np.clip(t, 0.1, 0.9)
        t = GuidedFilter(image * 255, 50, 0.001).filter(t * 255)

        A = estimateBackgroundLight(image)
        B = np.ones_like(image) * A

        cv2.imwrite(t_path, np.uint8(np.clip(t, 0, 255)))
        cv2.imwrite(b_path, np.uint8(B * 255))


def strip_module(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def load_bluenet():
    model = BLUE_Net(LayerNo=BLUENET_LAYERS)
    ckpt = torch.load(BLUENET_PATH, map_location=device)
    state = strip_module(ckpt["model_state_dict"])
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_ssuie():
    model = SS_UIE_model()
    state = torch.load(SSUIE_PATH, map_location=device)
    state = strip_module(state)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def tensor_to_bgr(t):
    img = t.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def compute_metrics(pred, gt):
    gt = cv2.resize(gt, IMG_SIZE)

    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    psnr = peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=255)
    ssim = structural_similarity(gt_rgb, pred_rgb, channel_axis=2, data_range=255)
    return psnr, ssim


def run_bluenet(model, x, t_prior, b_prior):
    with torch.inference_mode():
        outJ, _, _ = model(x, t_prior, b_prior)
    return outJ[-1]


def run_ssuie(model, x):
    with torch.inference_mode():
        return model(x)


def benchmark():
    for name, info in DATASETS.items():
        generate_priors(name, info)

    print("\nLoading models...")
    models = {"BlueNet": load_bluenet(), "SS-UIE": load_ssuie()}

    rows = []

    for model_name, model in models.items():
        for dname, info in DATASETS.items():
            input_dir = info["input"]
            gt_dir = info["gt"]
            base_dir = str(Path(input_dir).parent)

            t_dir = os.path.join(base_dir, "t_prior")
            b_dir = os.path.join(base_dir, "b_prior")

            save_path = os.path.join(SAVE_DIR, model_name, dname)
            os.makedirs(save_path, exist_ok=True)

            psnrs, ssims, times = [], [], []

            files = sorted(os.listdir(input_dir))

            print(f"\nRunning {model_name} on {dname}")

            for fname in tqdm(files):
                inp = cv2.imread(os.path.join(input_dir, fname))
                gtp = os.path.join(gt_dir, fname)

                if inp is None or not os.path.exists(gtp):
                    continue

                gt = cv2.imread(gtp)
                x = preprocess(inp)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()

                if model_name == "BlueNet":
                    stem = Path(fname).stem
                    tp = cv2.imread(os.path.join(t_dir, stem + ".png"), 0)
                    bp = cv2.imread(os.path.join(b_dir, stem + ".png"))

                    tp = cv2.resize(tp, IMG_SIZE)
                    bp = cv2.resize(bp, IMG_SIZE)

                    tp = (
                        torch.from_numpy(tp.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    tp = tp.repeat(1, 3, 1, 1).to(device)

                    bp = cv2.cvtColor(bp, cv2.COLOR_BGR2RGB)
                    bp = (
                        torch.from_numpy(bp.astype(np.float32) / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(device)
                    )

                    out = run_bluenet(model, x, tp, bp)

                else:
                    out = run_ssuie(model, x)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_time = time.time() - start
                times.append(infer_time)

                pred = tensor_to_bgr(out)
                cv2.imwrite(os.path.join(save_path, fname), pred)

                psnr, ssim = compute_metrics(pred, gt)
                psnrs.append(psnr)
                ssims.append(ssim)

            rows.append(
                {
                    "Model": model_name,
                    "Dataset": dname,
                    "PSNR": round(float(np.mean(psnrs)), 4),
                    "SSIM": round(float(np.mean(ssims)), 4),
                    "Time/Image (s)": round(float(np.mean(times)), 4),
                    "FPS": round(float(1 / max(np.mean(times), 1e-8)), 2),
                }
            )

    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(CSV_NAME, index=False)
    print(f"\nSaved results to {CSV_NAME}")
    plt.figure(figsize=(8, 5))
    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        plt.bar([f"{m}\n{dataset}" for m in sub["Model"]], sub["PSNR"])

    plt.title("PSNR Comparison")
    plt.ylabel("PSNR")
    plt.tight_layout()
    plt.savefig("psnr_plot.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        plt.bar([f"{m}\n{dataset}" for m in sub["Model"]], sub["SSIM"])

    plt.title("SSIM Comparison")
    plt.ylabel("SSIM")
    plt.tight_layout()
    plt.savefig("ssim_plot.png", dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset]
        plt.bar([f"{m}\n{dataset}" for m in sub["Model"]], sub["FPS"])

    plt.title("Speed Comparison (FPS)")
    plt.ylabel("FPS")
    plt.tight_layout()
    plt.savefig("fps_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    benchmark()
