import torch
import torch.nn as nn


class DR(nn.Module):
    def __init__(self, grid=8):
        super().__init__()
        self.mp = nn.AvgPool2d(grid, grid)
        self.T = nn.Tanh()

    def forward(self, x):
        d = self.mp(x).flatten(2, 3)
        ma, _ = d.max(2, keepdim=True)
        ma = ma.unsqueeze(3) + 0.0001
        mi, _ = d.min(2, keepdim=True)
        mi = mi.unsqueeze(3)
        return ma - mi


class CDR(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.dr = DR()

    def forward(self, x):
        return torch.pow(1 - self.dr(x), 5) + 0.05


class FAN(nn.Module):
    def __init__(self, grid=6):
        super().__init__()
        self.ap = nn.AvgPool2d(grid, grid)
        self.relu = nn.ReLU()

    def forward(self, x):
        df = self.ap(x)
        d = df.flatten(2, 3)
        ma, _ = d.max(2, keepdim=True)
        ma = ma.unsqueeze(3)
        mi, _ = d.min(2, keepdim=True)
        mi = mi.unsqueeze(3)
        x = (x - mi) / (ma - mi + 0.005)
        return torch.clip(x, 0, 1)


class DNnet(nn.Module):
    def __init__(self, dim=64, n=3):
        super(DNnet, self).__init__()
        self.n = n

        self.stem0 = nn.Sequential(
            nn.Conv2d(3 * pow(n, 2), dim, 3, 1, 1, padding_mode='replicate'),
            nn.Tanh()
        )
        self.out0 = nn.Sequential(
            nn.Conv2d(dim, 3 * pow(n, 2), 3, 1, 1, padding_mode='replicate'),

        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(3 * pow(n, 2), dim, 1, 1, padding_mode='replicate'),
            nn.Tanh()
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(dim, 3 * pow(n, 2), 1, 1, padding_mode='replicate'),
        )

        self.relu = nn.ReLU()
        self.fan = FAN()
        self.cdr = CDR()

        if n != 1:
            self.ps = nn.PixelShuffle(upscale_factor=n)
            self.pus = nn.PixelUnshuffle(downscale_factor=n)
        else:
            self.ps = nn.Identity()
            self.pus = nn.Identity()

    def forward(self, x):
        x0 = self.pus(x)
        x00 = self.stem0(x0)
        x11 = self.stem1(x0)
        c = self.cdr(x)
        xr = self.out0(self.relu(x00))
        xw = self.out1(torch.abs(x11))
        return self.fan(c * self.ps(xw * xr) + x)
