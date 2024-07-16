import torch.nn as nn

dropout = 0.4


class ResBlock(nn.Module):
    """
    Residual Block for ResNet.

    Args:
        block_size (int): Number of convolutional layers in the block.
        kernel_size (int): Size of the convolutional kernels.
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
    """
    def __init__(self, block_size: int, kernel_size: int, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super(ResBlock, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=stride, bias=False),
                                  nn.BatchNorm2d(out_ch), ) if stride > 1 else nn.Identity()
        self.act = nn.GELU()

        layers = [[] for _ in range(block_size)]
        self.blocks = []

        for i in range(block_size):
            if stride > 1 and i == 0:
                layers[i].append(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1, stride=stride, bias=False))
            else:
                layers[i].append(nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same', bias=False))

            layers[i].append(nn.BatchNorm2d(out_ch))
            layers[i].append(nn.GELU())
            layers[i].append(nn.Dropout2d(dropout, inplace=False))
            layers[i].append(nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same', bias=False))
            layers[i].append(nn.BatchNorm2d(out_ch))
            layers[i].append(nn.Dropout2d(dropout, inplace=False))

            layer_modules = nn.ModuleList(layers[i])
            self.blocks.append(nn.Sequential(*layer_modules))

        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        res = self.proj(x)

        for block in self.blocks:
            x = self.act(res + block(x))
            res = x

        return x


class ResNet34(nn.Module):
    """
    ResNet34 model.
    """
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, padding=1)

        self.block1 = ResBlock(3, 3, 64, 64, 1)
        self.block2 = ResBlock(4, 3, 64, 128, 2)
        self.block3 = ResBlock(6, 3, 128, 256, 2)
        self.block4 = ResBlock(3, 3, 256, 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, 1000)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = self.act(x)
        return x


class Model(nn.Module):
    """
    Custom model with ResNet34 backbone.

    Args:
        n_classes (int): Number of output classes.
        return_features (bool): Whether to return features or class scores.
    """
    def __init__(self, n_classes, return_features=False):
        super(Model, self).__init__()
        self.backbone = ResNet34()
        self.dense = nn.Linear(1000, n_classes)
        self.return_features = return_features
        self.embedding = None

    def forward(self, x):
        x = self.backbone(x)

        if self.return_features:
            return x

        self.embedding = x.detach()
        x = self.dense(x)
        return x
