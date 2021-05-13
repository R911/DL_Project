import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, num_channels, num_features, input_size=64, data_generation_mode=1):
        super(Discriminator, self).__init__()

        output_filter_size = 4
        if input_size is 32:
            output_filter_size = 2

        self.ngpu = ngpu
        self.data_generation_mode = data_generation_mode
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            # params: in_channels, out_channels, kernel_size, stride, padding
            nn.Conv2d(num_channels, num_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features) x 32 x 32
            nn.Conv2d(num_features, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*2) x 16 x 16
            nn.Conv2d(num_features * 2, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*4) x 8 x 8
            nn.Conv2d(num_features * 4, num_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (num_features*8) x 4 x 4
            nn.Conv2d(num_features * 8, 1, output_filter_size, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data):
        if self.training:
            return self.main(data)

        out = self.main[0](data)
        out1 = self.main[1](out)

        out2 = self.main[2](out1)
        out3 = self.main[3](out2)
        out4 = self.main[4](out3)

        out5 = self.main[5](out4)
        out6 = self.main[6](out5)
        out7 = self.main[7](out6)

        mp1 = nn.MaxPool2d(kernel_size=2, padding=0)
        mp2 = nn.MaxPool2d(kernel_size=2, padding=0)
        mp3 = nn.MaxPool2d(kernel_size=2, padding=0)

        if self.data_generation_mode is 1:
            flt1 = mp1(out1).flatten(start_dim=1)
            flt2 = mp2(out4).flatten(start_dim=1)
            flt3 = mp3(out7).flatten(start_dim=1)
        else:
            flt1 = mp1(out).flatten(start_dim=1)
            flt2 = mp2(out2).flatten(start_dim=1)
            flt3 = mp3(out5).flatten(start_dim=1)

        flt21 = torch.column_stack((flt1, flt2))
        final = torch.column_stack((flt21, flt3))

        return final
