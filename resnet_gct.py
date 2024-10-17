import torch.nn as nn
import torch
from torchvision.models import ResNet
import torch.nn.functional as F
from gct_module import Parametric_GCT


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# class GCT(nn.Module):
#     def __init__(self):
#         super(GCT, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.c = nn.Parameter(torch.tensor(0.0), requires_grad=True)

#     def forward(self, x):
#         # mean std
#         b, c, _, _ = x.size()
#         feature = self.avg_pool(x).reshape(b, c, -1)
#         mean_fea = torch.mean(feature, dim=1).unsqueeze(dim=1)
#         std_fea = torch.std(feature, dim=1).unsqueeze(dim=1)

#         norm_fea = (feature - mean_fea) / std_fea
#         gaussian_c = F.sigmoid(self.c) * 3 + 1
#         final_weights = torch.exp(-(norm_fea)**2 / (2*(gaussian_c)**2)).reshape(b, c, 1, 1)
#         return x*final_weights.expand_as(x)


class GCTBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(GCTBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = Parametric_GCT()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GCTBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(GCTBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = Parametric_GCT()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def gct_resnet18(num_classes=1000):
    """Constructs a ResNet-18 model."""

    model = ResNet(GCTBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gct_resnet34(num_classes=1000):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GCTBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gct_resnet50(num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GCTBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gct_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GCTBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gct_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GCTBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# class GCTLeNet(nn.Module):
#     def __init__(self, out_dim, init_weights=None):
#         super(GCTLeNet, self).__init__()
#         self.init_weights = init_weights
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1),
#             nn.BatchNorm2d(6),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#         )

#         self.gct =  GCT()

#         self.classifier = nn.Sequential(
#             nn.Linear(576, 120),  # in_features = 16 x6x6
#             nn.BatchNorm1d(120),
#             nn.ReLU(True),
#             nn.Linear(120, 84),
#             nn.BatchNorm1d(84),
#             nn.ReLU(True),
#             nn.Linear(84, out_dim),
#         )

#         if self.init_weights is not None:
#             self.init_weights_()

#     def forward(self, x):
#         a1 = self.feature_extractor(x)
#         a1 = self.gct(a1).reshape(x.shape[0], -1)
#         a2 = self.classifier(a1)
#         return a2
    

# class GCT_AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(GCT_AlexNet, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),

#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         )
        
#         self.fc = nn.Linear(256*6*6, num_classes)
#         self.gct_1 = GCT()
#         self.gct_2 = GCT()

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.gct_1(x)
#         x = self.conv2(x)
#         x = self.gct_2(x).reshape(x.shape[0], -1)
#         x = self.fc(x)
#         return x
