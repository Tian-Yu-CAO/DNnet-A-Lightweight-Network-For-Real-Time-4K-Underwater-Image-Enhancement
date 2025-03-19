from .SSIM_loss import *
from .color_column_loss import *


class combinedloss(nn.Module):
    def __init__(self,):
        super(combinedloss, self).__init__()
        self.colorcolumnloss = color_column_loss()

    def forward(self, out, label):
        ssim_loss = 1 - torch.mean(ssim(out, label))

        colorcolumn_loss_ = self.scatterloss(out, label)

        total_loss = ssim_loss+0.1*colorcolumn_loss_
        return total_loss



