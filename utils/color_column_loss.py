import torch.nn as nn
import torch


class color_column_loss(nn.Module):
    def __init__(self):
        super(color_column_loss, self).__init__()
        self.r = nn.ReLU()

    def forward(self, out, label):
        # 文中四个0.1宽度的bin两两相邻，这里直接计算宽度0.2的bin
        out_variants = torch.stack([self.r(-self.r(-out + i) + 0.2) for i in [0.1, 0.7]])
        label_variants = torch.stack([self.r(-self.r(-label + i) + 0.2) for i in [0.1, 0.7]])

        sub = (label_variants / (label_variants + 0.0001)).sum(dim=[2, 3]) - \
              (out_variants / (out_variants + 0.0001)).sum(
                  dim=[2, 3])
        sub = torch.pow(sub, 2)

        return sub.mean([0, 1, 2])/8000
