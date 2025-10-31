import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=2, Chans=2, Samples=750, dropoutRate=0.5, elu_alpha=1.0):
        super(DeepConvNet, self).__init__()
        self.elu_alpha = elu_alpha


        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding=0, bias=False),
            nn.Conv2d(25, 25, kernel_size=(Chans, 1), bias=False),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropoutRate)
        )


        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding=0, bias=False),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropoutRate)
        )


        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding=0, bias=False),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropoutRate)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding=0, bias=False),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
            nn.ELU(alpha=elu_alpha),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropoutRate)
        )


        self.classifier = None
        self.num_classes = num_classes

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1)

        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(x.shape[1], self.num_classes),
                nn.Softmax(dim=1)
            ).to(x.device)
        x = self.classifier(x)
        return x
