import numbers

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from numpy.random import RandomState
from scipy.stats import chi


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


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)

        return x


def quaternion_init(
    in_features, out_features, rng, kernel_size=None, criterion="glorot"
):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == "glorot":
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == "he":
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError("Invalid criterion: " + criterion)

    rng = RandomState(np.random.randint(1, 1234))

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4, loc=0, scale=s, size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    for i in range(0, number_of_weights):
        norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2 + 0.0001)
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i * np.sin(phase)
    weight_j = modulus * v_j * np.sin(phase)
    weight_k = modulus * v_k * np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)


def unitary_init(in_features, out_features, rng, kernel_size=None, criterion="he"):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i] ** 2 + v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
        v_r[i] /= norm
        v_i[i] /= norm
        v_j[i] /= norm
        v_k[i] /= norm
    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)


def random_init(in_features, out_features, rng, kernel_size=None, criterion="glorot"):
    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in = in_features * receptive_field
        fan_out = out_features * receptive_field
    else:
        fan_in = in_features
        fan_out = out_features

    if criterion == "glorot":
        s = 1.0 / np.sqrt(2 * (fan_in + fan_out))
    elif criterion == "he":
        s = 1.0 / np.sqrt(2 * fan_in)
    else:
        raise ValueError("Invalid criterion: " + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_i = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_j = np.random.uniform(-1.0, 1.0, number_of_weights)
    v_k = np.random.uniform(-1.0, 1.0, number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r
    weight_i = v_i
    weight_j = v_j
    weight_k = v_k
    return (weight_r, weight_i, weight_j, weight_k)


def get_kernel_and_weight_shape(operation, in_channels, out_channels, kernel_size):
    if operation == "convolution1d":
        if type(kernel_size) is not int:
            raise ValueError(str(kernel_size))
        else:
            ks = kernel_size
            w_shape = (out_channels, in_channels) + tuple((ks,))
    else:
        if operation == "convolution2d" and type(kernel_size) is int:
            ks = (kernel_size, kernel_size)
        elif operation == "convolution3d" and type(kernel_size) is int:
            ks = (kernel_size, kernel_size, kernel_size)
        elif type(kernel_size) is not int:
            if operation == "convolution2d" and len(kernel_size) != 2:
                raise ValueError(str(kernel_size))
            elif operation == "convolution3d" and len(kernel_size) != 3:
                raise ValueError(str(kernel_size))
            else:
                ks = kernel_size
        w_shape = (out_channels, in_channels) + (*ks,)
    return ks, w_shape


def affect_init_conv(
    r_weight, i_weight, j_weight, k_weight, kernel_size, init_func, rng, init_criterion
):
    if (
        r_weight.size() != i_weight.size()
        or r_weight.size() != j_weight.size()
        or r_weight.size() != k_weight.size()
    ):
        raise ValueError()

    elif 2 >= r_weight.dim():
        raise Exception()

    r, i, j, k = init_func(
        r_weight.size(1),
        r_weight.size(0),
        rng=rng,
        kernel_size=kernel_size,
        criterion=init_criterion,
    )
    r, i, j, k = (
        torch.from_numpy(r),
        torch.from_numpy(i),
        torch.from_numpy(j),
        torch.from_numpy(k),
    )
    r_weight.data = r.type_as(r_weight.data)
    i_weight.data = i.type_as(i_weight.data)
    j_weight.data = j.type_as(j_weight.data)
    k_weight.data = k.type_as(k_weight.data)


def quaternion_conv_rotation(
    input,
    zero_kernel,
    r_weight,
    i_weight,
    j_weight,
    k_weight,
    bias,
    stride,
    padding,
    groups,
    dilatation,
    quaternion_format,
    scale=None,
):
    square_r = r_weight * r_weight
    square_i = i_weight * i_weight
    square_j = j_weight * j_weight
    square_k = k_weight * k_weight

    norm = torch.sqrt(square_r + square_i + square_j + square_k + 0.0001)

    r_n_weight = r_weight / norm
    i_n_weight = i_weight / norm
    j_n_weight = j_weight / norm
    k_n_weight = k_weight / norm

    norm_factor = 2.0

    square_i = norm_factor * (i_n_weight * i_n_weight)
    square_j = norm_factor * (j_n_weight * j_n_weight)
    square_k = norm_factor * (k_n_weight * k_n_weight)

    ri = norm_factor * r_n_weight * i_n_weight
    rj = norm_factor * r_n_weight * j_n_weight
    rk = norm_factor * r_n_weight * k_n_weight

    ij = norm_factor * i_n_weight * j_n_weight
    ik = norm_factor * i_n_weight * k_n_weight

    jk = norm_factor * j_n_weight * k_n_weight

    if quaternion_format:
        if scale is not None:
            rot_kernel_1 = torch.cat(
                [
                    zero_kernel,
                    scale * (1.0 - (square_j + square_k)),
                    scale * (ij - rk),
                    scale * (ik + rj),
                ],
                dim=1,
            )
            rot_kernel_2 = torch.cat(
                [
                    zero_kernel,
                    scale * (ij + rk),
                    scale * (1.0 - (square_i + square_k)),
                    scale * (jk - ri),
                ],
                dim=1,
            )
            rot_kernel_3 = torch.cat(
                [
                    zero_kernel,
                    scale * (ik - rj),
                    scale * (jk + ri),
                    scale * (1.0 - (square_i + square_j)),
                ],
                dim=1,
            )
        else:
            rot_kernel_1 = torch.cat(
                [zero_kernel, (1.0 - (square_j + square_k)), (ij - rk), (ik + rj)],
                dim=1,
            )
            rot_kernel_2 = torch.cat(
                [zero_kernel, (ij + rk), (1.0 - (square_i + square_k)), (jk - ri)],
                dim=1,
            )
            rot_kernel_3 = torch.cat(
                [zero_kernel, (ik - rj), (jk + ri), (1.0 - (square_i + square_j))],
                dim=1,
            )

        zero_kernel2 = torch.cat(
            [zero_kernel, zero_kernel, zero_kernel, zero_kernel], dim=1
        )
        global_rot_kernel = torch.cat(
            [zero_kernel2, rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0
        )

    else:
        if scale is not None:
            rot_kernel_1 = torch.cat(
                [
                    scale * (1.0 - (square_j + square_k)),
                    scale * (ij - rk),
                    scale * (ik + rj),
                ],
                dim=0,
            )
            rot_kernel_2 = torch.cat(
                [
                    scale * (ij + rk),
                    scale * (1.0 - (square_i + square_k)),
                    scale * (jk - ri),
                ],
                dim=0,
            )
            rot_kernel_3 = torch.cat(
                [
                    scale * (ik - rj),
                    scale * (jk + ri),
                    scale * (1.0 - (square_i + square_j)),
                ],
                dim=0,
            )
        else:
            rot_kernel_1 = torch.cat(
                [1.0 - (square_j + square_k), (ij - rk), (ik + rj)], dim=0
            )
            rot_kernel_2 = torch.cat(
                [(ij + rk), 1.0 - (square_i + square_k), (jk - ri)], dim=0
            )
            rot_kernel_3 = torch.cat(
                [(ik - rj), (jk + ri), (1.0 - (square_i + square_j))], dim=0
            )

        global_rot_kernel = torch.cat([rot_kernel_1, rot_kernel_2, rot_kernel_3], dim=0)

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(str(input.dim()))

    return convfunc(input, global_rot_kernel, bias, stride, padding, dilatation, groups)


def quaternion_conv(
    input,
    r_weight,
    i_weight,
    j_weight,
    k_weight,
    bias,
    stride,
    padding,
    groups,
    dilatation,
):
    cat_kernels_4_r = torch.cat([r_weight, -i_weight, -j_weight, -k_weight], dim=1)
    cat_kernels_4_i = torch.cat([i_weight, r_weight, -k_weight, j_weight], dim=1)
    cat_kernels_4_j = torch.cat([j_weight, k_weight, r_weight, -i_weight], dim=1)
    cat_kernels_4_k = torch.cat([k_weight, -j_weight, i_weight, r_weight], dim=1)

    cat_kernels_4_quaternion = torch.cat(
        [cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0
    )

    if input.dim() == 3:
        convfunc = F.conv1d
    elif input.dim() == 4:
        convfunc = F.conv2d
    elif input.dim() == 5:
        convfunc = F.conv3d
    else:
        raise Exception(str(input.dim()))

    return convfunc(
        input, cat_kernels_4_quaternion, bias, stride, padding, dilatation, groups
    )


class QuaternionConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilatation=1,
        padding=0,
        groups=1,
        bias=True,
        init_criterion="glorot",
        weight_init="quaternion",
        seed=None,
        operation="convolution2d",
        rotation=True,
        quaternion_format=True,
        scale=False,
    ):
        super(QuaternionConv, self).__init__()

        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = RandomState(self.seed)
        self.operation = operation
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.winit = {
            "quaternion": quaternion_init,
            "unitary": unitary_init,
            "random": random_init,
        }[self.weight_init]
        self.scale = scale

        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape(
            self.operation, self.in_channels, self.out_channels, kernel_size
        )

        self.r_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.i_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.j_weight = nn.Parameter(torch.Tensor(*self.w_shape))
        self.k_weight = nn.Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param = nn.Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = nn.Parameter(
                torch.zeros(self.r_weight.shape), requires_grad=False
            )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(
            self.r_weight,
            self.i_weight,
            self.j_weight,
            self.k_weight,
            self.kernel_size,
            self.winit,
            self.rng,
            self.init_criterion,
        )
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, inp):
        if self.rotation:
            return quaternion_conv_rotation(
                inp,
                self.zero_kernel,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
                self.stride,
                self.padding,
                self.groups,
                self.dilatation,
                self.quaternion_format,
                self.scale_param,
            )
        else:
            return quaternion_conv(
                inp,
                self.r_weight,
                self.i_weight,
                self.j_weight,
                self.k_weight,
                self.bias,
                self.stride,
                self.padding,
                self.groups,
                self.dilatation,
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_channels="
            + str(self.in_channels)
            + ", out_channels="
            + str(self.out_channels)
            + ", bias="
            + str(self.bias is not None)
            + ", kernel_size="
            + str(self.kernel_size)
            + ", stride="
            + str(self.stride)
            + ", padding="
            + str(self.padding)
            + ", init_criterion="
            + str(self.init_criterion)
            + ", weight_init="
            + str(self.weight_init)
            + ", seed="
            + str(self.seed)
            + ", rotation="
            + str(self.rotation)
            + ", q_format="
            + str(self.quaternion_format)
            + ", operation="
            + str(self.operation)
            + ")"
        )


class FeedForward(nn.Module):
    def __init__(self, dim_in, dim_out, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim_in * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim_in, hidden_features * 2, kernel_size=1, bias=bias
        )

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class SelfAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=1,
        ffn_expansion_factor=2.66,
        bias=True,
        LayerNorm_type="WithBias",
    ):
        super(SelfAttentionTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, k, nonlinear="relu"):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.k = k
        self.nonlinear = nonlinear

        self.linear1 = nn.Linear(channels, channels // k)
        self.linear2 = nn.Linear(channels // k, channels)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        if nonlinear == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == "PReLU":
            self.activation = nn.PReLU()
        else:
            raise ValueError

    def attention(self, x):
        N, C, H, W = x.size()
        out = torch.flatten(self.global_pooling(x), 1)
        out = self.activation(self.linear1(out))
        out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)

        return out.mul(x)

    def forward(self, x):
        return self.attention(x)


class Aggreation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        k=8,
        nonlinear="PReLU",
        norm="in",
    ):
        super(Aggreation, self).__init__()
        self.attention = SelfAttention(in_channels, k, nonlinear="relu")
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            nonlinear=nonlinear,
            norm=norm,
        )

    def forward(self, x):
        res = self.conv(self.attention(x))
        return res


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        groups=1,
        norm="in",
        nonlinear="relu",
    ):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.norm = norm
        self.nonlinear = nonlinear

        if norm == "bn":
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.normalization = nn.InstanceNorm2d(out_channels, affine=False)
        elif norm == "ln":
            self.normalization = LayerNorm(dim=out_channels, LayerNorm_type="BiasFree")
        else:
            self.normalization = None

        if nonlinear == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == "PReLU":
            self.activation = nn.PReLU()
        elif nonlinear == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))

        if self.normalization is not None:
            out = self.normalization(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class SPP(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_layers=4, interpolation_type="bilinear"
    ):
        super(SPP, self).__init__()
        self.conv = nn.ModuleList()
        self.num_layers = num_layers
        self.interpolation_type = interpolation_type

        for _ in range(self.num_layers):
            self.conv.append(
                ConvLayer(
                    in_channels,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    nonlinear="PReLU",
                    norm=None,
                )
            )

        self.fusion = ConvLayer(
            (in_channels * (self.num_layers + 1)),
            out_channels,
            kernel_size=3,
            stride=1,
            norm="False",
            nonlinear="PReLU",
        )

    def forward(self, x):
        N, C, H, W = x.size()
        out = []

        for level in range(self.num_layers):
            out.append(
                F.interpolate(
                    self.conv[level](
                        F.avg_pool2d(
                            x,
                            kernel_size=2 * 2 ** (level + 1),
                            stride=2 * 2 ** (level + 1),
                            padding=2 * 2 ** (level + 1) % 2,
                        )
                    ),
                    size=(H, W),
                    mode=self.interpolation_type,
                )
            )

        out.append(x)

        return self.fusion(torch.cat(out, dim=1))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * num_heads * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim * num_heads, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim * num_heads,
            dim * num_heads,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * num_heads,
            bias=bias,
        )
        self.kv_dwconv = nn.Conv2d(
            dim * num_heads * 2,
            dim * num_heads * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * num_heads * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim * num_heads, dim, kernel_size=1, bias=bias)

    def forward(self, q, k):
        b, c, h, w = k.shape

        kv = self.kv_dwconv(self.kv(k))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(q))

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out)
        return out


class CrossAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=2,
        ffn_expansion_factor=2.66,
        bias=True,
        layerNorm_type="WithBias",
    ):
        super(CrossAttentionTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, layerNorm_type)
        self.norm2 = LayerNorm(dim, layerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)

        self.norm3 = LayerNorm(dim, layerNorm_type)
        self.ffn = FeedForward(dim, dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        y = y + self.attn(self.norm1(x), self.norm2(y))
        y = y + self.ffn(self.norm3(y))

        return y


class MAQ(nn.Module):
    def __init__(self, dim):
        super(MAQ, self).__init__()
        self.branch1 = CrossAttentionTransformer(dim)
        self.branch2 = CrossAttentionTransformer(dim)
        self.branch3 = SelfAttentionTransformer(dim)
        self.qcnn = nn.Sequential(
            QuaternionConv(
                in_channels=dim * 4,
                out_channels=dim * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.SiLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x, prior):
        x1 = self.branch1(prior, x)
        x2 = self.branch2(x, prior)
        x3 = self.branch3(x)

        z = torch.zeros_like(x, device=x.device)
        out = torch.cat((z, x1, x2, x3), 1)
        out = self.qcnn(out)
        out = self.final(out)
        return out


class Condition(nn.Module):
    def __init__(self, in_nc=3, nf=32):
        super(Condition, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out


class ScaleHarmonizer(nn.Module):
    def __init__(self, in_nc=6, out_nc=3, base_nf=64, cond_nf=32):
        super(ScaleHarmonizer, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_scale3 = nn.Linear(cond_nf, out_nc, bias=True)

        self.cond_shift1 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift2 = nn.Linear(cond_nf, base_nf, bias=True)
        self.cond_shift3 = nn.Linear(cond_nf, out_nc, bias=True)

        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = self.cond_net(x)

        scale1 = self.cond_scale1(cond)
        shift1 = self.cond_shift1(cond)

        scale2 = self.cond_scale2(cond)
        shift2 = self.cond_shift2(cond)

        scale3 = self.cond_scale3(cond)
        shift3 = self.cond_shift3(cond)

        out = self.conv1(x)
        out = (
            out * scale1.view(-1, self.base_nf, 1, 1)
            + shift1.view(-1, self.base_nf, 1, 1)
            + out
        )
        out = self.act(out)
        out = self.conv2(out)
        out = (
            out * scale2.view(-1, self.base_nf, 1, 1)
            + shift2.view(-1, self.base_nf, 1, 1)
            + out
        )
        out = self.act(out)
        out = self.conv3(out)
        out = (
            out * scale3.view(-1, self.out_nc, 1, 1)
            + shift3.view(-1, self.out_nc, 1, 1)
            + out
        )

        return out


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class FeatureContextualizer(nn.Module):
    def __init__(self, ch_in=3, dim=16, ch_out=6, prior_ch=3):
        super(FeatureContextualizer, self).__init__()

        self.embed = OverlapPatchEmbed(ch_in, dim)
        self.embed_prior = OverlapPatchEmbed(prior_ch, dim)

        self.block1_1 = MAQ(dim)
        self.block1_2 = MAQ(dim)
        self.agg1 = Aggreation(dim * 2, dim)

        self.block2_1 = MAQ(dim)
        self.block2_2 = MAQ(dim)
        self.agg2 = Aggreation(dim * 3, dim)

        self.spp = SPP(dim, ch_out)

    def forward(self, x, prior):
        x = self.embed(x)
        prior_embed = self.embed_prior(prior)

        x_1 = self.block1_1(x, prior_embed)
        x_2 = self.block1_2(x_1, x_1)
        x1 = self.agg1(torch.cat((x_1, x_2), dim=1))

        x_1 = self.block2_1(x1, prior_embed)
        x_2 = self.block2_2(x_1, x_1)
        x2 = self.agg2(torch.cat((x1, x_1, x_2), dim=1))

        out = self.spp(x2)

        return out


class ColorBalancePrior(nn.Module):
    def __init__(self, ch_in=3):
        super(ColorBalancePrior, self).__init__()
        self.enc = NAFBlock(ch_in)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_mean = x_mean.expand_as(x)

        prior = self.enc(x_mean)

        return prior


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
            mat_inv = torch.linalg.inv(mat.float().contiguous())
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
        print("Loading subnetworks .....")
        Z_Net = [NAFBlock(3), NAFBlock(3), NAFBlock(3)]
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
        ).to(J.device)
        J_m_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 1], 1), 1), 1
        ).to(J.device)
        J_s_bar = torch.unsqueeze(
            torch.unsqueeze(torch.unsqueeze(list_mean_sorted[:, 0], 1), 1), 1
        ).to(J.device)
        J_m_bar = J_l - u + (1.0 / lambda_1) * w_1
        J_s_bar = J_l - v + (1.0 / lambda_2) * w_2
        J_l = J_l.to(J.device)
        J_m = J_m.to(J.device)
        J_s = J_s.to(J.device)
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
        D = torch.ones(I.shape, device=I.device)
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
        self.fc = FeatureContextualizer(
            ch_in=n_feat + scale_unetfeats * 3,
            dim=48,
            ch_out=n_feat + scale_unetfeats * 3,
            prior_ch=3,
        )

    def forward(self, x2_img, stage1_img, feat1, res1, x2_samfeats, cbp):
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)
        feat2, feat_fin2 = self.stage2_encoder(x2_cat, feat1, res1)
        cbp_down = F.interpolate(
            cbp, size=feat_fin2.shape[2:], mode="bilinear", align_corners=False
        )
        feat_fin2 = self.fc(feat_fin2, cbp_down)
        res2 = self.stage2_decoder(feat_fin2, feat2)
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)
        return x3_samfeats, stage2_img, feat2, res2


class Model(torch.nn.Module):
    def __init__(self, LayerNo=5):
        super(Model, self).__init__()
        self.LayerNo = LayerNo
        net_layers = []
        for i in range(LayerNo):
            net_layers.append(BasicBlock())
        self.uunet = nn.ModuleList(net_layers)
        n_feat = 40
        scale_unetfeats = 20
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
        self.color_prior = ColorBalancePrior(ch_in=3)
        self.final_harmonizer = ScaleHarmonizer(
            in_nc=6, out_nc=3, base_nf=64, cond_nf=32
        )

    def forward(self, I, t_p, B_p):
        bs, _, _, _ = I.shape
        cbp = self.color_prior(I)
        B = torch.zeros((bs, 3, 1, 1), device=I.device)
        t = torch.zeros(I.shape, device=I.device)
        J = I.to(I.device)
        Y = torch.zeros(I.shape, device=I.device)
        Z = torch.zeros(I.shape, device=I.device)
        Q = torch.zeros(I.shape, device=I.device)
        R = torch.zeros(I.shape, device=I.device)
        u = torch.zeros((bs, 1, 1, 1), device=I.device)
        v = torch.zeros((bs, 1, 1, 1), device=I.device)
        w_1 = torch.zeros((bs, 1, 1, 1), device=I.device)
        w_2 = torch.zeros((bs, 1, 1, 1), device=I.device)
        list_output = []
        list_B = []
        list_t = []
        beta = torch.tensor([3.001], device=I.device)
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
                img, stage1_img, feat1, res1, x2_samfeats, cbp
            )
            Y = stage1_img
            if j < self.LayerNo - 1:
                list_output.append(J.detach())
                list_B.append(B.detach())
                list_t.append(t.detach())
            else:
                harmonizer_input = torch.cat([J, cbp], dim=1)
                J_refined = self.final_harmonizer(harmonizer_input)
                list_output.append(J_refined)
                list_B.append(B)
                list_t.append(t)
        return list_output, list_B, list_t
