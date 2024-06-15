import torch
import torch.nn as nn


class CSPUpBlock(nn.Module):
    """
    CSPUpBlock class, implementation from https://github.com/Skyfallk/2024_deep_gen_models/tree/main/HW_2.GAN_train
    """

    def __init__(self, in_channels):
        super(CSPUpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv3_1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

    def forward(self, x):
        # 1 way
        # TODO: test
        split_x = torch.split(x, 512, dim=1)
        # conv0_x = self.conv1(x)
        upsampled_x = self.upsample(split_x[0])

        # 2 way
        # conv1_x = self.conv1(x)
        relu_x = self.relu(split_x[1])
        upsampled_conv1_x = self.upsample1(relu_x)
        conv3_x1 = self.conv3_1(upsampled_conv1_x)
        relu_conv3_x1 = self.relu(conv3_x1)
        conv3_x2 = self.conv3_2(relu_conv3_x1)

        # merge
        out = upsampled_x + conv3_x2

        return out


class Generator(nn.Module):
    """
    CSPUpBlock class, implementation from https://github.com/Skyfallk/2024_deep_gen_models/tree/main/HW_2.GAN_train
    """

    def __init__(self, in_channels=1024):
        super(Generator, self).__init__()

        self.norm = nn.ConvTranspose2d(100, in_channels, 4, 1, 0, bias=False)
        self.csp_up_block1 = CSPUpBlock(in_channels=in_channels)
        self.csp_up_block2 = CSPUpBlock(in_channels=in_channels // 2)
        self.csp_up_block3 = CSPUpBlock(in_channels=in_channels // 4)
        self.csp_up_block4 = CSPUpBlock(in_channels=in_channels // 8)

        self.deconv2d = nn.ConvTranspose2d(in_channels // 16, 3, kernel_size=2, stride=2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.norm(x)
        x = self.csp_up_block1(x)
        x = self.csp_up_block2(x)
        x = self.csp_up_block3(x)
        x = self.csp_up_block4(x)
        x = self.deconv2d(x)
        y = self.relu(x)
        return y


class Discriminator(nn.Module):
    """
    Discriminator custom
    """

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    print("Encoder row")
    x = torch.randn(1, 1024, 4, 4)

    split_x = torch.split(x, 512, dim=1)
    print(split_x)
    # csp_up_block = CSPUpBlock(in_channels=1024)
    # out = csp_up_block(x)
    #
    # print("Output shape:", out.shape)
    #
    # csp_up_block = CSPUpBlock(in_channels=512)
    # out = csp_up_block(out)
    #
    # print("Output shape:", out.shape)
    #
    # csp_up_block = CSPUpBlock(in_channels=256)
    # out = csp_up_block(out)
    #
    # print("Output shape:", out.shape)
    #
    # csp_up_block = CSPUpBlock(in_channels=128)
    # out = csp_up_block(out)
    #
    # print("Output shape:", out.shape)
    #
    # deconv2d = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
    # relu = nn.ReLU(inplace=True)
    #
    # out = deconv2d(out)
    # out = relu(out)
    #
    # print("Output shape:", out.shape)
    #
    # print("Decoder row")
    #
    # print(Generator())
    # print(Discriminator())
