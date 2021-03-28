import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: weight init

class Z_pereodic_gen(nn.Module):
    def __init__(self, opt):
        super(Z_pereodic_gen, self).__init__()
        self.opt = opt

        self.l = nn.Linear(in_features=self.opt.global_noise_dim, out_features=self.opt.hidden_noise_dim)
        self.l1 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)
        self.l2 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)

    def forward(self, Z_g):
        x = self.l(Z_g)
        x = F.relu(x)
        K1 = self.l1(x)
        K2 = self.l2(x)

        Z_p = torch.zeros(self.opt.batch_size, self.periodic_noise_dim, self.spatial_size, self.spatial_size)
        # I think it can be better
        for l in range(self.opt.spatial_size):
            for m in range(self.opt.spatial_size):
                Z_p[:, :, l, m] = k1*l + k2*m

        phi = torch.rand(self.opt.batch_size, self.opt.periodic_noise_dim, 1, 1) * 2.0 * math.pi
        Z_p = torch.sin(Z_p + phi)

        return Z_p

class PSGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Generator, self).__init__()
        self.opt = opt
        noise_dim = self.opt.local_noise_dim + self.opt.global_noise_dim + self.opt.periodic_noise_dim

        # Z_p generator
        self.z_p_gen = Z_pereodic_gen(opt)

        # Generator layers
        layers = []
        # Repiting layers
        for in_c, out_c in zip([noise_dim] + self.opt.gen_conv_channels[:-2], self.opt.gen_conv_channels[:-1]):
            layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=self.opt.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
        # Last layer
        layers.append(nn.ConvTranspose2d(in_channels=self.opt.gen_conv_channels[-2], out_channels=self.opt.gen_conv_channels[-1], kernel_size=self.opt.kernel_size, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)

    def forward(self, Z_l, Z_g):
        assert Z_l.shape[1] == self.opt.local_noise_dim
        assert Z_l.shape[2] == self.opt.local_noise_dim
        assert Z_g.shape[1] == self.opt.global_noise_dim
        # Z pereodic
        Z_p = self.z_p_gen(Z_g)
        # Summarized Z
        Z = torch.cat((Z, Z_p), dim=1)

        x = self.gen(Z)
        return x

class PSGAN_Discriminator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Discriminator, self).__init__()
        self.opt = opt
        noise_dim = self.opt.local_noise_dim + self.opt.global_noise_dim + self.opt.periodic_noise_dim

        # Generator layers
        layers = []
        # Repiting layers
        for in_c, out_c in zip(self.opt.dis_conv_channels[:-2], self.opt.dis_conv_channels[1:-1]):
            layers.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=self.opt.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        # Remove first BN
        # TODO: Do we really need this?
        layers.pop(1)
        # Last layer
        layers.append(nn.Conv2d(in_channels=self.opt.dis_conv_channels[-2], out_channels=self.opt.dis_conv_channels[-1], kernel_size=self.opt.kernel_size, stride=2, padding=1))
        layers.append(nn.Sigmoid())

        self.dis = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dis(x)
        x = x.view(self.opt.batch_size, -1)
        return x
