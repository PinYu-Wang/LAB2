import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes=2, Chans=2, Samples=750,
                 F1=16, D=2, kernel_len=64, dropoutRate=0.25, elu_alpha=1.0):
        super(EEGNet, self).__init__()
        self.num_classes = num_classes
        self.elu_alpha = elu_alpha
        self.classify = None

        # -------- FirstConv (temporal conv) --------
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_len), stride=(1, 1),
                      padding=(0, kernel_len // 2), bias=False),
            nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha)
        )

        # -------- Depthwise Conv block --------
        self.depthwiseBlock = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1), stride=(1, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=dropoutRate)
        )

        # -------- Separable Conv block --------
        self.separableBlock = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=dropoutRate)
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseBlock(x)
        x = self.separableBlock(x)
        x = x.flatten(1)

        if self.classify is None:
            in_features = x.shape[1]
            self.classify = nn.Linear(in_features, self.num_classes).to(x.device)

        x = self.classify(x)
        return x
