import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import interpolate, conv

class BayesModel(nn.Module):

    def __init__(self, in_channels=1024, c_channels=256, alpha=0.8, beta=0.8, pixel_weighting=None, device=None, layer=None):
        super().__init__()

        self.project = conv(in_channels, c_channels, 1, bias=False)
        self.alpha = alpha
        self.beta = beta
        self.n_fg = 0
        self.n_bg = 0
        self.s_fg = 0
        self.s_bg = 0

        self.pw_params = pixel_weighting
        self.device = device
        self.layer = layer

    def _is_finite(self, t):
        return (torch.isnan(t) + torch.isinf(t)) == 0

    def compute_pixel_weights(self, y):
        """
        :param pixel_weighting:   dict(method={'fixed'|'hinged'}, tf=target influence (fraction), per_frame=bool)
        :param y:                 Training labels (tensor, shape = (N, 1, H, W)), N = number of samples
        :return:  tensor (N, 1, H, W) of pixel weights
        """
        if self.pw_params is None or self.pw_params['method'] == 'none':
            return torch.ones_like(y)

        method = self.pw_params['method']
        tf = self.pw_params['tf']  # Target influence of foreground pixels
        assert method == 'hinge'

        N, C, H, W = y.shape

        y = y.float()
        px = y.sum(dim=(2, 3))
        af = px / (H * W)
        px = px.view(N, C, 1, 1)
        af = af.view(N, C, 1, 1)

        too_small = (px < 10).float()
        af = too_small * tf + (1 - too_small) * af

        ii = (af > tf).float()        # if af > tf (i.e the object is large enough), then set tf = af
        tf = ii * af + (1 - ii) * tf  # this leads to self.pixel_weighting = 1 (i.e do nothing)

        wf = tf / af  # Foreground pixels weight
        wb = (1 - tf) / (1 - af)  # Background pixels weight

        training = False
        if training:
            invalid = ~self._is_finite(wf)
            if invalid.sum().item() > 0:
                print("Warning: Corrected invalid (non-finite) foreground filter pixel-weights.")
                wf[invalid] = 1.0

            invalid = ~self._is_finite(wb)
            if invalid.sum() > 0:
                print("Warning: Corrected bad (non-finite) background pixel-weights.")
                wb[invalid] = 1.0

        w = wf * y + wb * (1 - y)
        w = torch.sqrt(w)

        return w

    def distance(self, x, s):
        batch_size, nchannels, h, w = x.size() 
        delta = x - s.view(batch_size, nchannels, 1, 1)
        delta = delta.permute(0, 2, 3, 1).reshape(-1, 1, n_channels)
        mm = torch.bmm(delta, delta.transpose(1, 2))
        dist = 2 / (1 + torch.exp(mm.view(batch_size, h, w)))
        return dist

    def init(self, x, y):
        batch_size, nchannels, h, w = x.size()
        pw = self.compute_pixel_weights(y)
        x = self.project(x)
        y = F.interpolate(y, (h, w), mode="nearest")
        fg = x * y
        bg = x * (1 - y)        
        self.n_fg = torch.sum(y.view(16, -1), dim=1)
        self.n_bg = torch.sum((1 - y).view(16, -1), dim=1)
        self.s_fg = fg.view(16, 1024, -1).sum(dim=2) / self.n_fg.unsqueeze(1)
        self.s_bg = bg.view(16, 1024, -1).sum(dim=2) / self.n_bg.unsqueeze(1)

    def forward(self, x):
        batch_size, nchannels, h, w = x.size()
        x = self.project(x)
        prior_fg = self.n_fg / (self.n_fg + self.n_bg)
        prior_bg = 1 - prior_fg
        ll_fg = self.distance(x, self.s_fg)
        ll_bg = self.distance(x, self.s_bg)
        post_fg = prior_fg.view(batch_size, 1, 1) * ll_fg / (prior_fg.view(batch_size, 1, 1) * ll_fg + prior_bg.view(batch_size, 1, 1) * ll_bg)
        post_bg = prior_bg.view(batch_size, 1, 1) * ll_bg / (prior_fg.view(batch_size, 1, 1) * ll_fg + prior_bg.view(batch_size, 1, 1) * ll_bg)
        nt_fg = post_fg.view(batch_size, -1).sum(dim=1)
        nt_bg = post_bg.view(batch_size, -1).sum(dim=1)
        self.n_fg = self.alpha * self.n_fg + nt_fg
        self.n_bg = self.alpha * self.n_bg + nt_bg
        fg = x * post_fg.unsqueeze(1)
        bg = x * post_bg.unsqueeze(1)
        self.s_fg = self.beta * self.s_fg + fg.view(batch_size, nchannels, -1).sum(dim=2) / nt_fg
        self.s_bg = self.beta * self.s_bg + bg.view(batch_size, nchannels, -1).sum(dim=2) / nt_bg
        return post_fg.unsqueeze(1)