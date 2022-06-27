import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_ratio=0.5):
        super(DoubleConv, self).__init__()
        mlist = []
        mlist.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        mlist.append(nn.ELU())
        mlist.append(nn.Dropout(dropout_ratio))
        mlist.append(
            nn.Conv2d(
                out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False
            )
        )
        mlist.append(nn.ELU())
        self.conv = nn.Sequential(*mlist)

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64, 128],
        dropout_ratios=[0.5] * 5
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
                in_channels, feature, dropout_ratio=p))
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
            self.ups.append(DoubleConv(feature * 2, feature, dropout_ratio=p))

        self.bottleneck = DoubleConv(
            features[-1], features[-1] * 2, dropout_ratio=bottleneck_p)
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


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    print(model)
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
