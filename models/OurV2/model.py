import math
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat


def _t(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float32)


def _mul(coeffs, image):
    coeffs = coeffs.to(image.device).view(3, 3, 1, 1)
    return torch.nn.functional.conv2d(image, coeffs)


_RGB_TO_XYZ = _t(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
WHITE_POINTS = {"d65": _t([0.9504, 1.0000, 1.0888]).view(1, 3, 1, 1)}
_XYZ_TO_LAB = _t([[0.0, 116.0, 0.0], [500.0, -500.0, 0.0], [0.0, 200.0, -200.0]])
_LAB_OFF = _t([16.0, 0.0, 0.0]).view(1, 3, 1, 1)


def remove_gamma(rgb):
    T = 0.04045
    rgb1 = torch.max(rgb, rgb.new_tensor(T))
    return torch.where(
        rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4)
    )


def _lab_f(x):
    epsilon = 0.008856
    kappa = 903.3
    x1 = torch.max(x, x.new_tensor(epsilon))
    return torch.where(x > epsilon, torch.pow(x1, 1.0 / 3), (kappa * x + 16.0) / 116.0)


def rgb2lab(rgb):
    rgb = remove_gamma(rgb)
    xyz = _mul(_RGB_TO_XYZ, rgb)
    xyz = xyz / WHITE_POINTS["d65"].to(xyz.device)
    f_xyz = _lab_f(xyz)
    return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF.to(xyz.device)


def rgb2lch(rgb):
    lab = rgb2lab(rgb)
    l = lab[:, 0, :, :]
    c = torch.norm(lab[:, 1:, :, :], 2, 1)
    h = torch.atan2(lab[:, 2, :, :], lab[:, 1, :, :]) * (180 / 3.14159)
    h = torch.where(h >= 0, h, 360 + h)
    return torch.stack([l, c, h], 1)


def batch_PSNR(img, gt, data_range=1.0):
    img, gt = img.clamp(0, data_range), gt.clamp(0, data_range)
    mse = torch.mean((img - gt) ** 2, dim=(1, 2, 3))
    return 10 * torch.log10(data_range**2 / (mse + 1e-8))


def quantAB(bins, vmax, vmin, device):
    a = torch.linspace(
        vmin + ((vmax - vmin) / (bins * 2)),
        vmax - ((vmax - vmin) / (bins * 2)),
        bins,
        device=device,
    )
    return torch.cartesian_prod(a, a).view(1, bins**2, 2, 1, 1)


class lab_Loss(nn.Module):
    def __init__(self, alpha=1, weight=1, levels=7, vmin=-80, vmax=80):
        super().__init__()
        self.alpha, self.weight, self.levels, self.vmin, self.vmax = (
            alpha,
            weight,
            levels,
            vmin,
            vmax,
        )

    def forward(self, img, gt):
        tab = quantAB(self.levels, self.vmax, self.vmin, img.device)
        l_img, l_gt = (
            torch.clamp(rgb2lab(img), self.vmin, self.vmax),
            torch.clamp(rgb2lab(gt), self.vmin, self.vmax),
        )
        p = torch.nn.functional.softmax(
            -self.alpha * ((l_img[:, 1:, :, :].unsqueeze(1) - tab) ** 2).sum(2), dim=1
        )
        q = torch.nn.functional.softmax(
            -self.alpha * ((l_gt[:, 1:, :, :].unsqueeze(1) - tab) ** 2).sum(2), dim=1
        )
        return (
            torch.abs(l_img[:, 0] - l_gt[:, 0]).mean()
            - self.weight
            * (q * torch.log(torch.clamp(p, 0.001, 0.999))).sum([1, 2, 3]).mean()
        )


class lch_Loss(nn.Module):
    def __init__(self, weightC=1, weightH=1, levels=4):
        super().__init__()
        self.wC, self.wH, self.lvls = weightC, weightH, levels

    def forward(self, img, gt):
        i_lch, g_lch = rgb2lch(img), rgb2lch(gt)
        h = i_lch[:, 2] * (self.lvls / 360.0)
        a = torch.arange(self.lvls).float().to(h.device).view(1, self.lvls, 1, 1)
        p = (torch.relu(1 - torch.abs(h.unsqueeze(1) - a)) + 0.01) / (
            1.0 + self.lvls * 0.01
        )
        gh = g_lch[:, 2] * (self.lvls / 360.0)
        q = torch.relu(1 - torch.abs(gh.unsqueeze(1) - a))
        return (
            torch.abs(i_lch[:, 0] - g_lch[:, 0]).mean()
            + self.wC * torch.abs(i_lch[:, 1] - g_lch[:, 1]).mean()
            + self.wH
            * torch.mean(-torch.mul(q, torch.log(torch.clamp(p, 0.001, 0.999))))
        )


class FDL(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.w = weight

    def forward(self, p, t):
        return (
            self.w
            * torch.pow(
                torch.abs(
                    torch.fft.rfft2(p, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
                ),
                2,
            ).mean()
        )


class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.ws = window_size
        self.channel = 3
        window = self.create_window(window_size, self.channel)
        self.register_buffer("window", window)

    def create_window(self, ws, ch):
        g = torch.Tensor(
            [exp(-((x - ws // 2) ** 2) / float(2 * 1.5**2)) for x in range(ws)]
        )
        g /= g.sum()
        return (
            (g.unsqueeze(1).mm(g.unsqueeze(1).t()))
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(ch, 1, ws, ws)
            .contiguous()
        )

    def forward(self, i1, i2):
        mu1, mu2 = (
            F.conv2d(i1, self.window, padding=self.ws // 2, groups=self.channel),
            F.conv2d(i2, self.window, padding=self.ws // 2, groups=self.channel),
        )
        s1, s2, s12 = (
            F.conv2d(i1 * i1, self.window, padding=self.ws // 2, groups=self.channel)
            - mu1**2,
            F.conv2d(i2 * i2, self.window, padding=self.ws // 2, groups=self.channel)
            - mu2**2,
            F.conv2d(i1 * i2, self.window, padding=self.ws // 2, groups=self.channel)
            - mu1 * mu2,
        )
        return (
            ((2 * mu1 * mu2 + 0.0001) * (2 * s12 + 0.0009))
            / ((mu1**2 + mu2**2 + 0.0001) * (s1 + s2 + 0.0009))
        ).mean(dim=(1, 2, 3))


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


class Model(nn.Module):
    def __init__(self, c=32, nm=3, nr=3, H=256, W=256):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(3, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
        )

        self.e2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, 1, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1),
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, 1, 1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.Conv2d(c * 4, c * 4, 3, 1, 1),
        )

        self.p = nn.MaxPool2d(2, 2)
        self.m = nn.ModuleList(
            [MemoryBlock(c * 4, nr, i + 1, H // 4, W // 4) for i in range(nm)]
        )

        self.fuse_attn = nn.Sequential(
            nn.Conv2d(c * 4 * nm, c * 4 * nm, 1), nn.Sigmoid()
        )

        self.f = nn.Conv2d(c * 4 * nm, c * 4, 3, 1, 1)
        self.up1 = nn.Conv2d(c * 4, c * 2, 3, 1, 1)
        self.up2 = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.skip1 = nn.Conv2d(c * 2, c * 2, 1)
        self.skip2 = nn.Conv2d(c, c, 1)

        self.out = nn.Conv2d(c, 3, 3, 1, 1)

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        r0 = x

        x1 = self.e1(x)
        x2 = self.p(self.e2(x1))
        x3 = self.p(self.e3(x2))

        ys = [x3]
        feats = []

        for b in self.m:
            x3 = b(x3, ys)
            feats.append(x3)

        feat_cat = torch.cat(feats, dim=1)
        attn = self.fuse_attn(feat_cat)
        p = self.f(feat_cat * attn)

        p = F.interpolate(p, scale_factor=2)
        p = self.up1(p) + self.skip1(x2)

        p = F.interpolate(p, scale_factor=2)
        p = self.up2(p) + self.skip2(x1)

        out = self.out(p)

        return r0 + self.alpha * out
