import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_K(nn.Module):
    def __init__(self, opt):
        super(MLP_K, self).__init__()
        self.opt = opt

        self.l = nn.Linear(in_features=self.opt.global_noise_dim, out_features=self.opt.hidden_noise_dim)
        self.l1 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)
        self.l2 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)

    def forward(self, Z_g):
        x = self.l(Z_g)
        x = F.relu(x)
        K1 = self.l1(x)
        K2 = self.l2(x)

        return (K1, K2)

class PSGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Generator, self).__init__()
        self.opt = opt
        noise_dim = self.opt.local_noise_dim + self.opt.global_noise_dim + self.opt.periodic_noise_dim

        # Generator layers
        layers = []

        for in_c, out_c in zip([noise_dim] + self.opt.conv_channels[:-2], self.opt.conv_channels[:-1]):
            layers.append(nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=self.opt.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(in_channels=self.opt.conv_channels[-2], out_channels=self.opt.conv_channels[-1], kernel_size=self.opt.kernel_size, stride=2, padding=1))
        layers.append(nn.Tanh())

        self.gen = nn.Sequential(*layers)