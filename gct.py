import torch.nn as nn
import torch
from torchvision.models import ResNet
import torch.nn.functional as F


class Non_Parameteric_GCT(nn.Module): 
    '''Parameter Free version of GCT (denoted by GCT-B0 in official paper)'''
    def __init__(self):
        super(Non_Parameteric_GCT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c = 2 # constant value

    def forward(self, x):
        # mean std
        b, c, _, _ = x.size()
        feature = self.avg_pool(x).reshape(b, c, -1)
        mean_fea = torch.mean(feature, dim=1).unsqueeze(dim=1)
        std_fea = torch.std(feature, dim=1).unsqueeze(dim=1)

        norm_fea = (feature - mean_fea) / std_fea
        gaussian_c = F.sigmoid(self.c) * 3 + 1
        final_weights = torch.exp(-(norm_fea)**2 / (2*(gaussian_c)**2)).reshape(b, c, 1, 1)
        return x*final_weights.expand_as(x)
    

class Parametric_GCT(nn.Module):
    '''Parameteric version of GCT (denoted by GCT-B1 in official paper)'''
    def __init__(self):
        super(Parametric_GCT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c = nn.Parameter(torch.tensor(0.0), requires_grad=True) # parameter - just 1 param

    def forward(self, x):
        # mean std
        b, c, _, _ = x.size()
        feature = self.avg_pool(x).reshape(b, c, -1)
        mean_fea = torch.mean(feature, dim=1).unsqueeze(dim=1)
        std_fea = torch.std(feature, dim=1).unsqueeze(dim=1)

        norm_fea = (feature - mean_fea) / std_fea
        gaussian_c = F.sigmoid(self.c) * 3 + 1
        final_weights = torch.exp(-(norm_fea)**2 / (2*(gaussian_c)**2)).reshape(b, c, 1, 1)
        return x*final_weights.expand_as(x)