import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, kl
from torchvision.models.vgg import VGG16_Weights, vgg16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU())
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Compute_z(nn.Module):
    def __init__(self, latent_dim):
        super(Compute_z, self).__init__()
        self.latent_dim = latent_dim
        self.u_conv_layer = nn.Conv2d(
            128, 2 * self.latent_dim, kernel_size=1, padding=0
        )
        self.s_conv_layer = nn.Conv2d(
            128, 2 * self.latent_dim, kernel_size=1, padding=0
        )

    def forward(self, x):
        u_encoding = torch.mean(x, dim=2, keepdim=True)
        u_encoding = torch.mean(u_encoding, dim=3, keepdim=True)
        u_mu_log_sigma = self.u_conv_layer(u_encoding)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu_log_sigma = torch.squeeze(u_mu_log_sigma, dim=2)
        u_mu = u_mu_log_sigma[:, : self.latent_dim]
        u_log_sigma = u_mu_log_sigma[:, self.latent_dim :]
        u_dist = Independent(Normal(loc=u_mu, scale=torch.exp(u_log_sigma)), 1)

        s_encoding = torch.std(x, dim=2, keepdim=True)
        s_encoding = torch.std(s_encoding, dim=3, keepdim=True)
        s_mu_log_sigma = self.s_conv_layer(s_encoding)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu_log_sigma = torch.squeeze(s_mu_log_sigma, dim=2)
        s_mu = s_mu_log_sigma[:, : self.latent_dim]
        s_log_sigma = s_mu_log_sigma[:, self.latent_dim :]
        s_dist = Independent(Normal(loc=s_mu, scale=torch.exp(s_log_sigma)), 1)
        return (
            u_dist,
            s_dist,
            u_mu,
            s_mu,
            torch.exp(u_log_sigma),
            torch.exp(s_log_sigma),
        )


class Encoder(nn.Module):
    def __init__(self, ch):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=64)
        self.conv3 = ConvBlock(ch_in=64, ch_out=64)
        self.conv4 = ResBlock(64)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        x4 = self.conv4(x4)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.pr_encoder = Encoder(3)
        self.po_encoder = Encoder(6)

        self.pr_conv = ResBlock(64)
        self.po_conv = ResBlock(64)

        self.pr_Up3 = Up()
        self.pr_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up2 = Up()
        self.pr_UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up1 = Up()
        self.pr_UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.po_Up3 = Up()
        self.po_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.po_Up2 = Up()
        self.po_UpConv2 = ConvBlock(ch_in=128, ch_out=64)

        out_conv = []
        out_conv.append(ResBlock(64))
        out_conv.append(ResBlock(64))
        out_conv.append(nn.Conv2d(64, 3, kernel_size=1, padding=0))
        self.out_conv = nn.Sequential(*out_conv)

        z = 20

        self.compute_z_pr = Compute_z(z)
        self.compute_z_po = Compute_z(z)

        self.conv_u = nn.Conv2d(z, 128, kernel_size=1, padding=0)
        self.conv_s = nn.Conv2d(z, 128, kernel_size=1, padding=0)

        self.insnorm = nn.InstanceNorm2d(128)
        self.sigmoid = nn.Sigmoid()

    def forward(self, Input):

        pr_x1, pr_x2, pr_x3, pr_x4 = self.pr_encoder.forward(Input)

        # x4->x3
        pr_x4 = self.pr_conv(pr_x4)
        pr_d3 = self.pr_Up3(pr_x4)
        # x3->x2
        pr_d3 = torch.cat((pr_x3, pr_d3), dim=1)
        pr_d3 = self.pr_UpConv3(pr_d3)
        pr_d2 = self.pr_Up2(pr_d3)
        # x2->x1
        pr_d2 = torch.cat((pr_x2, pr_d2), dim=1)
        pr_d2 = self.pr_UpConv2(pr_d2)
        pr_d1 = self.pr_Up1(pr_d2)
        # cat
        pr_d1 = torch.cat((pr_x1, pr_d1), dim=1)
        # x1->dis
        pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
        pr_latent_u = pr_u_dist.rsample()
        pr_latent_s = pr_s_dist.rsample()
        pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
        pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
        pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
        pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
        pr_u = self.conv_u(pr_latent_u)
        pr_s = self.conv_s(pr_latent_s)
        pr_d1 = self.insnorm(pr_d1) * torch.abs(pr_s) + pr_u
        # x1->out
        pr_d1 = self.pr_UpConv1(pr_d1)
        out = self.out_conv(pr_d1)
        return out


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(
            self.loss_network(out_images), self.loss_network(target_images)
        )
        return perception_loss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = Decoder().to(DEVICE)
        self.criterion = nn.MSELoss().to(DEVICE)
        self.VGG16 = PerceptionLoss().to(DEVICE)

    def forward(self, Input):
        self.Input = Input
        self.out = self.decoder.forward(self.Input)
        return self.out

    def kl_divergence(self, analytic=True):
        if analytic:
            kl_div_u = torch.mean(kl.kl_divergence(self.po_u, self.pr_u))
            kl_div_s = torch.mean(kl.kl_divergence(self.po_s, self.pr_s))
        return kl_div_u + kl_div_s

    def elbo(self, target, analytic_kl=True):
        self.kl_loss = self.kl_divergence(analytic=analytic_kl)
        self.reconstruction_loss = self.criterion(self.out, target)
        self.vgg16_loss = self.VGG16(self.out, target)
        return self.reconstruction_loss + self.vgg16_loss + self.kl_loss
