import torch
from torch import nn
from torch.nn import functional as F
from lib.utils import conv, relu, interpolate, adaptive_cat


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()

        nc = ic + oc
        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(), conv(nc, nc, 3), relu(), conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):
        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.conv2 = conv(in_channels // 2, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2 * x.shape[-2], 2 * x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class PyrUpBicubic2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        def kernel(d):
            x = d + torch.arange(-1, 3, dtype=torch.float32)
            x = torch.abs(x)
            a = -0.75
            f = (x < 1).float() * ((a + 2) * x * x * x - (a + 3) * x * x + 1) + \
                ((x >= 1) * (x < 2)).float() * (a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a)
            W = f.reshape(1, 1, 1, len(x)).float()
            Wt = W.permute(0, 1, 3, 2)
            return W, Wt

        We, We_t = kernel(-0.25)
        Wo, Wo_t = kernel(-0.25 - 0.5)

        # Building non-separable filters for now. It would make sense to
        # have separable filters if it proves to be faster.

        # .contiguous() is needed until a bug is fixed in nn.Conv2d.
        self.W00 = (We_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W01 = (We_t @ Wo).expand(channels, 1, 4, 4).contiguous()
        self.W10 = (Wo_t @ We).expand(channels, 1, 4, 4).contiguous()
        self.W11 = (Wo_t @ Wo).expand(channels, 1, 4, 4).contiguous()

    def forward(self, input):

        if input.device != self.W00.device:
            self.W00 = self.W00.to(input.device)
            self.W01 = self.W01.to(input.device)
            self.W10 = self.W10.to(input.device)
            self.W11 = self.W11.to(input.device)

        a = F.pad(input, (2, 2, 2, 2), 'replicate')

        I00 = F.conv2d(a, self.W00, groups=self.channels)
        I01 = F.conv2d(a, self.W01, groups=self.channels)
        I10 = F.conv2d(a, self.W10, groups=self.channels)
        I11 = F.conv2d(a, self.W11, groups=self.channels)

        n, c, h, w = I11.shape

        J0 = torch.stack((I00, I01), dim=-1).view(n, c, h, 2 * w)
        J1 = torch.stack((I10, I11), dim=-1).view(n, c, h, 2 * w)
        out = torch.stack((J0, J1), dim=-2).view(n, c, 2 * h, 2 * w)

        out = F.pad(out, (-1, -1, -1, -1))
        return out


class BackwardCompatibleUpsampler(nn.Module):
    """ Upsampler with bicubic interpolation that works with Pytorch 1.0.1 """

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels // 2, 3)
        self.up1 = PyrUpBicubic2d(in_channels)
        self.conv2 = conv(in_channels // 2, 1, 3)
        self.up2 = PyrUpBicubic2d(in_channels // 2)

    def forward(self, x, image_size):
        x = self.up1(x)
        x = F.relu(self.conv1(x))
        x = self.up2(x)
        x = F.interpolate(x, image_size[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(x)
        return x


class BaseUp(nn.Module):
    
    def __init__(self, f_channels, s_channels, out_channels, bias=False):
        super().__init__()
        self.conv1 = conv(s_channels+f_channels, out_channels, 3, bias)
        self.conv2 = conv(out_channels, out_channels, 3, bias)

    def forward(self, f, s):
        h, w = f.shape[-2:]
        s = interpolate(s, (h, w))
        x = self.conv1(torch.cat((f, s), dim=1))
        x = self.conv2(x)
        return x


class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        # self.TSE = nn.ModuleDict()
        # self.RRB1 = nn.ModuleDict()
        # self.CAB = nn.ModuleDict()
        # self.RRB2 = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        self.up5 = BaseUp(2048, 1, oc)
        self.up4 = BaseUp(1024, oc, oc)
        self.up3 = BaseUp(512, oc, oc)
        self.up2 = BaseUp(256, oc, oc)


        # for L, fc in self.ft_channels.items():
        #     self.TSE[L] = TSE(fc, ic, oc)
        #     self.RRB1[L] = RRB(oc, use_bn=use_bn)
        #     self.CAB[L] = CAB(oc, L == 'layer5')
        #     self.RRB2[L] = RRB(oc, use_bn=use_bn)

        #if torch.__version__ == '1.0.1'
        self.project = BackwardCompatibleUpsampler(out_channels)
        #self.project = Upsampler(out_channels)

    def forward(self, scores, features, image_size):

        # x = None
        # for i, L in enumerate(self.ft_channels):
        #     ft = features[L]
        #     s = interpolate(scores, ft.shape[-2:])  # Resample scores to match features size

        #     h, hpool = self.TSE[L](ft, s, x)
        #     h = self.RRB1[L](h)
        #     h = self.CAB[L](hpool, h)
        #     x = self.RRB2[L](h)

        x = self.up5(features['layer5'], scores)
        x = self.up4(features['layer4'], x)
        x = self.up3(features['layer3'], x)
        x = self.up2(features['layer2'], x)

        x = self.project(x, image_size)
        return x
