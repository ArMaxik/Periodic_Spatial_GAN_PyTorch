import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Generator, self).__init__()
        self.opt = opt
        noise_dim = self.opt.image_coder_dim + self.opt.local_noise_dim + self.opt.global_noise_dim + self.opt.periodic_noise_dim

        # Z_p generator
        self.l = nn.Linear(in_features=self.opt.global_noise_dim, out_features=self.opt.hidden_noise_dim)
        self.l1 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)
        self.l2 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)

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
        # Coder
        if self.opt.image_coder_dim > 0:
            self._init_coder()

        self.apply(weights_init)

    def _init_coder(self):
        layers = []
        # Repiting layers
        for in_c, out_c in zip(self.opt.dis_conv_channels[:-2], self.opt.dis_conv_channels[1:-1]):
            layers.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=self.opt.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        # layers.pop(1)
        # Last layer
        layers.append(nn.Conv2d(in_channels=self.opt.dis_conv_channels[-2], out_channels=self.opt.image_coder_dim, kernel_size=self.opt.kernel_size, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(self.opt.image_coder_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.cod = nn.Sequential(*layers)
        self.cod_l1 = nn.Linear(in_features=self.opt.image_coder_dim * self.opt.spatial_size * self.opt.spatial_size, out_features = self.opt.image_coder_dim*self.opt.spatial_size)
        self.cod_leaky = nn.LeakyReLU(negative_slope=0.2)
        self.cod_l2 = nn.Linear(in_features=self.opt.image_coder_dim*self.opt.spatial_size, out_features = self.opt.image_coder_dim)
        self.cod_tanh = nn.Tanh()


    def _expand_Z_c(self, Z_c, spatial_size):
        Z_c = Z_c.view(Z_c.shape[0], -1, 1, 1)
        pad = (
            spatial_size // 2 - 1 + spatial_size % 2,
            spatial_size // 2,
            spatial_size // 2 - 1 + spatial_size % 2,
            spatial_size // 2
        )
        Z_c = F.pad(Z_c, pad, mode='replicate')
        return Z_c

    def forward(self, Z_l, Z_g, imgs, spatial_size=None):
        if spatial_size == None:
            spatial_size=self.opt.spatial_size
        assert Z_l.shape[1] == self.opt.local_noise_dim
        # assert Z_g.shape[1] == self.opt.global_noise_dim
        # Z coder
        Z_cat = []
        if self.opt.image_coder_dim > 0:
            Z_c = self.cod(imgs)
            Z_c = self.cod_l1(Z_c.view(Z_c.shape[0], -1))
            Z_c = self.cod_leaky(Z_c)
            Z_c = self.cod_l2(Z_c)
            Z_c = self.cod_tanh(Z_c)
            Z_c = self._expand_Z_c(Z_c, spatial_size)
            Z_cat.append(Z_c)
        # Z local
        # Z_l = self.cod(imgs)
        # Z global
        # Z_g = self.cod(imgs)
        # Z_g = self._expand_Z_g(Z_g)
        # Z pereodic
        Z_p = self._z_p_gen(Z_g, spatial_size)
        Z_cat.extend([Z_l, Z_g, Z_p])
        # Summarized Z
        Z = torch.cat(Z_cat, dim=1)

        x = self.gen(Z)
        return x

    def _z_p_gen(self, Z_g, spatial_size):
        current_batch_size = Z_g.shape[0]
        Z_g = Z_g[:,:,0, 0].view(current_batch_size, self.opt.global_noise_dim)
        x = self.l(Z_g)
        x = F.relu(x)
        K1 = self.l1(x)
        K2 = self.l2(x)

        Z_p = torch.zeros(current_batch_size, self.opt.periodic_noise_dim, spatial_size, spatial_size, device=Z_g.device)
        # I think it can be better
        for l in range(spatial_size):
            for m in range(spatial_size):
                Z_p[:, :, l, m] = K1*l + K2*m

        phi = torch.rand(current_batch_size, self.opt.periodic_noise_dim, 1, 1, device=Z_g.device) * 2.0 * math.pi
        Z_p = torch.sin(Z_p + phi)

        return Z_p

class PSGAN_Discriminator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Discriminator, self).__init__()
        self.opt = opt

        # Discriminator layers
        self.layers = nn.ModuleList()
        # Repiting layers
        for in_c, out_c in zip(self.opt.dis_conv_channels[:-2], self.opt.dis_conv_channels[1:-1]):
            self.layers.append(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=self.opt.kernel_size, stride=2, padding=1))
            self.layers.append(nn.BatchNorm2d(out_c))
            self.layers.append(nn.LeakyReLU(negative_slope=0.2))
        # layers.pop(1)
        # Last layer
        self.layers.append(nn.Conv2d(in_channels=self.opt.dis_conv_channels[-2], out_channels=self.opt.dis_conv_channels[-1], kernel_size=self.opt.kernel_size, stride=2, padding=1))
        self.layers.append(nn.Sigmoid())

        # self.dis = nn.Sequential(*layers)

        self.apply(weights_init)

    def forward(self, x):
        # x = self.dis(x)
        style_activations = []
        for layer in self.layers:
            x = layer(x)
            style_activations.append(x)

        x = x.view(x.shape[0], -1)
        return (x, style_activations)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        c = torch.rand(1).item()*math.pi
        nn.init.normal_(m.bias.data, c, 0.02*c)
