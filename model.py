from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def get_activation(act):
    if act == 'elu':
        activation = nn.ELU()
    elif act == 'relu':
        activation = nn.ReLU()
    elif act == 'lrelu':
        activation = nn.LeakyReLU(0.1)
    else:
        raise NotImplementedError(f'{act} is not supported yet')
    return activation


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dropout_ratio=0.5,
        use_bn=False,
        act='elu'
    ):
        super(DoubleConv, self).__init__()
        mlist = []
        mlist.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        if use_bn:
            mlist.append(nn.BatchNorm2d(out_channels))
        mlist.append(get_activation(act))
        mlist.append(nn.Dropout(dropout_ratio))
        mlist.append(
            nn.Conv2d(
                out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        if use_bn:
            mlist.append(nn.BatchNorm2d(out_channels))
        mlist.append(get_activation(act))
        self.conv = nn.Sequential(*mlist)

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, chs) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(chs, chs, 1, 1)
        self.conv2 = nn.Conv2d(chs, chs, 1, 1)
        self.conv3 = nn.Conv2d(chs, 1, 1, 1)
        self.norm = nn.BatchNorm2d(chs)
        self.norm1 = nn.BatchNorm2d(1)

    def forward(self, x, shortcut):
        g = self.norm(self.conv1(shortcut))
        x = self.norm(self.conv2(x))
        psi = F.elu(g+x)
        psi = self.norm1(self.conv3(psi))
        psi = torch.sigmoid(psi)
        return torch.matmul(x, psi)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64, 128],
        dropout_ratios=[0.5] * 5,
        use_bn=False,
        act='elu'
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        dropouts = dropout_ratios[:4]
        bottleneck_p = dropout_ratios[-1]

        # Down part of UNET
        for feature, p in zip(features, dropouts):
            self.downs.append(DoubleConv(
                in_channels, feature, dropout_ratio=p, use_bn=use_bn, act=act))
            in_channels = feature

        # Up part of UNET
        for feature, p in zip(reversed(features), reversed(dropouts)):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature,
                            dropout_ratio=p, use_bn=use_bn, act=act))

        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2, dropout_ratio=bottleneck_p, use_bn=use_bn, act=act)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.sigmoid(self.final_conv(x))


class AttenUNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64, 128],
        dropout_ratios=[0.5] * 5,
        use_bn=False,
        act='elu'
    ):
        super(AttenUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        dropouts = dropout_ratios[:4]
        bottleneck_p = dropout_ratios[-1]

        # Down part of UNET
        for feature, p in zip(features, dropouts):
            self.downs.append(DoubleConv(
                in_channels, feature, dropout_ratio=p, use_bn=use_bn, act=act))
            in_channels = feature

        # Up part of UNET
        for feature, p in zip(reversed(features), reversed(dropouts)):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(AttentionBlock(feature))
            self.ups.append(DoubleConv(feature * 2, feature,
                            dropout_ratio=p, use_bn=use_bn, act=act))

        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2, dropout_ratio=bottleneck_p, use_bn=use_bn, act=act)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 3):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 3]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            atten = self.ups[idx + 1](x, skip_connection)
            concat_skip = torch.cat((x, atten), dim=1)
            x = self.ups[idx + 2](concat_skip)

        return self.sigmoid(self.final_conv(x))


def test():
    x = torch.randn((3, 1, 161, 161))
    model = AttenUNET(in_channels=1, out_channels=1)
    print(model)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
