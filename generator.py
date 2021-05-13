import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, num_channels, num_features, latent_vect_size, output_size=64):
        super(Generator, self).__init__()
        final_stride = 2

        if output_size is 64:
            final_stride = 2
        elif output_size is 32:
            final_stride = 1

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_vect_size, num_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # state size. (num_features*8) x 4 x 4
            nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # state size. (num_features*4) x 8 x 8
            nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # state size. (num_features*2) x 16 x 16
            nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # state size. (num_features) x 32 x 32
            nn.ConvTranspose2d(num_features, num_channels, 4, final_stride, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, data):
        return self.main(data)
