import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import interpolate, conv


class LayerCascadeModule(nn.Module):

    def __init__(self, cur_channels, prev_channels=None, next_channels=None, em_channels=64, out_channels=256):
        super().__init__()

        self.cur_conv1 = conv(cur_channels, em_channels, 1, bias=False)
        self.cur_conv2 = conv(em_channels, em_channels, 1, bias=False)
        self.prev_conv = conv(prev_channels, em_channels, 1, bias=False) if prev_channels else None
        self.next_conv = conv(next_channels, em_channels, 1, bias=False) if next_channels else None
        valid = (prev_channels is not None) + (next_channels is not None)
        self.out_conv = conv(cur_channels + valid * em_channels, out_channels, 1, bias=False)

    def forward(self, cur, pre, nxt):
        b, c, h, w = cur.size()
        s = self.cur_conv1(cur)
        s = F.adaptive_max_pool2d(s, (1, 1))
        s = self.cur_conv2(s)
        s = F.sigmoid(s)
        if self.prev_conv:
            pre = interpolate(pre, (h, w))
            pre = self.prev_conv(pre)
            pre *= s
            cur = torch.cat((pre, cur), dim=1)
        if self.next_conv:
            nxt = interpolate(nxt, (h, w))
            nxt = self.next_conv(nxt)
            nxt *= s
            cur = torch.cat((cur, nxt), dim=1)
        ret = self.out_conv(cur)
        return ret


if __name__ == "__main__":
    net = LayerCascadeModule(512, 256, None)
    pre = torch.rand(1, 256, 120, 216)
    cur = torch.rand(1, 512, 60, 108)
    nxt = torch.rand(1, 1024, 30, 54)
    print(net(cur, pre, None).shape)