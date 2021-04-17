import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

# =================================================
# Custom layers
# =================================================
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

class EqualConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(EqualConv2d, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5

        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x

class EqualConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super(EqualConvTranspose2d, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5

        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x

class EqualLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(EqualLinear, self).__init__(*args, **kwargs)
        self.c = (2 / self.weight.data[0].numel()) ** 0.5

        nn.init.normal_(self.weight.data, 0.0, 1.0)
        nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        x = super().forward(x) * self.c
        return x

# =================================================
# Networks
# =================================================

class PSGAN_Generator(nn.Module):
    def __init__(self, opt):
        super(PSGAN_Generator, self).__init__()
        self.opt = opt
        noise_dim = self.opt.image_coder_dim + self.opt.local_noise_dim + self.opt.global_noise_dim + self.opt.periodic_noise_dim

        # Z_p generator
        self.l = nn.Linear(in_features=self.opt.global_noise_dim, out_features=self.opt.hidden_noise_dim)
        self.l1 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)
        self.l2 = nn.Linear(in_features=self.opt.hidden_noise_dim, out_features=self.opt.periodic_noise_dim)

        # Init layer iterator
        self.cur_layer_iter = iter(self.opt.gen_conv_channels)
        self.cur_layer_dim = next(self.cur_layer_iter)

        self.layers = nn.ModuleList([
            EqualConvTranspose2d(noise_dim, self.cur_layer_dim, kernel_size=self.opt.kernel_size, stride=2, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ])
        self.toRGB = EqualConv2d(self.cur_layer_dim, 3, kernel_size=(1, 1), bias=True)


        self._init_coder()


    def _init_coder(self):
        # Coder layers
        self.cod_layers = nn.ModuleList([
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, self.opt.image_coder_dim, kernel_size=self.opt.kernel_size, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
        ])
        self.cod_fromRGB = EqualConv2d(3, self.cur_layer_dim, (1, 1), bias=True)
        
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

    def add_block(self):
        # Generator
        pre_layer_dim = self.cur_layer_dim
        self.cur_layer_dim = next(self.cur_layer_iter)
        block = nn.ModuleList([
            nn.Upsample(scale_factor=2.0),
            EqualConv2d(pre_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            PixelNorm(),
            nn.LeakyReLU(0.2),
        ])
        self.block_size = len(block)
        self.toRGB_new = EqualConv2d(self.cur_layer_dim, 3, (1, 1), bias=True)
        self.layers.extend(block)

        # Coder
        cod_block = nn.ModuleList([
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, pre_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        ])
        self.cod_block_size = len(cod_block)
        self.cod_fromRGB_new = EqualConv2d(3, self.cur_layer_dim, (1, 1), bias=True)
        
        self.cod_layers = cod_block.extend(self.cod_layers)

    def _coder_forward(self, imgs, spatial_size, alpha = -1):
        # No trasition
        if alpha == -1:
            Z_c = self.cod_fromRGB(imgs)
            for layer in self.cod_layers:
                Z_c = layer(Z_c)
        # Transition
        else:
            Z_c_old = torch.nn.functional.avg_pool2d(imgs, kernel_size = 2)
            Z_c_old = self.cod_fromRGB(Z_c_old)

            Z_c_new = self.cod_fromRGB_new(imgs)
            for layer in self.cod_layers[:self.cod_block_size]: 
                Z_c_new = layer(Z_c_new)
            
            Z_c = Z_c_new * alpha + Z_c_old * (1.0 - alpha)
            for layer in self.cod_layers[self.cod_block_size:]:
                Z_c = layer(Z_c)

        # Fully conected layers
        Z_c = self.cod_l1(Z_c.view(Z_c.shape[0], -1))
        Z_c = self.cod_leaky(Z_c)
        Z_c = self.cod_l2(Z_c)
        Z_c = self.cod_tanh(Z_c)
        Z_c = self._expand_Z_c(Z_c, spatial_size)
        return Z_c

    def forward(self, Z_l, Z_g, imgs, alpha = -1, spatial_size=None):
        if spatial_size == None:
            spatial_size=self.opt.spatial_size
        assert Z_l.shape[1] == self.opt.local_noise_dim

        # Z coder
        Z_c = self._coder_forward(imgs, spatial_size=spatial_size, alpha=alpha)
        # Z local
        # Z_l = self.cod(imgs)
        # Z global
        # Z_g = self.cod(imgs)
        # Z_g = self._expand_Z_g(Z_g)
        # Z pereodic
        Z_p = self._z_p_gen(Z_g, spatial_size)
        # Summarized Z
        # print("\n"*3, Z_c.shape, Z_l.shape, Z_g.shape, Z_p.shape,"\n"*3)
        x = torch.cat((Z_c, Z_l, Z_g, Z_p), dim=1)

        if not alpha == -1:
            return self.transition_forward(x, alpha)
        
        return self.normal_forward(x)

    def normal_forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.toRGB(x)
        x = torch.tanh(x)    
        return x

    def transition_forward(self, x, alpha):
        for layer in self.layers[:-self.block_size]:
            x = layer(x)

        x_old = nn.functional.interpolate(x, size = x.shape[2] * 2)
        x_old = self.toRGB(x_old)
        x_old = torch.tanh(x_old)

        x_new = x
        for layer in self.layers[-self.block_size:]:
            x_new = layer(x_new)
        x_new = self.toRGB_new(x_new)
        x_new = torch.tanh(x_new)

        x = x_new * alpha + x_old * (1.0 - alpha)

        return x

    def end_transition(self):
        self.cod_fromRGB = self.cod_fromRGB_new
        self.toRGB = self.toRGB_new

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
        # Init layer iterator
        self.cur_layer_iter = iter(self.opt.gen_conv_channels)
        self.cur_layer_dim = next(self.cur_layer_iter)
        
        # Discriminator layers
        self.layers = nn.ModuleList([
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=self.opt.kernel_size, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2),
        ])
        self.fromRGB = EqualConv2d(3, self.cur_layer_dim, (1, 1), bias=True)
        self.lrelu_fromRGB = nn.LeakyReLU(0.2)

    def add_block(self):
        pre_layer_dim = self.cur_layer_dim
        self.cur_layer_dim = next(self.cur_layer_iter)

        block = nn.ModuleList([
            EqualConv2d(self.cur_layer_dim, self.cur_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            EqualConv2d(self.cur_layer_dim, pre_layer_dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        ])
        self.block_size = len(block)
        self.fromRGB_new = EqualConv2d(3, self.cur_layer_dim, (1, 1), bias=True)
        
        self.layers = block.extend(self.layers)

    def forward(self, x, alpha = -1):
        if not alpha == -1:
            return self.transition_forward(x, alpha)
        
        return self.normal_forward(x) 

    def normal_forward(self, x):
        x = self.fromRGB(x)
        x = self.lrelu_fromRGB(x)

        for layer in self.layers:
            x = layer(x)

        return x

    def transition_forward(self, x, alpha):
        x_old = torch.nn.functional.avg_pool2d(x, kernel_size = 2)
        x_old = self.fromRGB(x_old)
        x_old = self.lrelu_fromRGB(x_old)

        x_new = self.fromRGB_new(x)
        x_new = self.lrelu_fromRGB(x_new)
        for layer in self.layers[:self.block_size]: 
            x_new = layer(x_new)
        
        x = x_new * alpha + x_old * (1.0 - alpha)

        for layer in self.layers[self.block_size:]:
            x = layer(x)

        return x

    def end_transition(self):
        self.fromRGB = self.fromRGB_new

    # def forward(self, x):
    #     # x = self.dis(x)
    #     for layer in self.layers[:-2]:
    #         x = layer(x)
    #     adv_loss_x = x
    #     for layer in self.layers[-2:]:
    #         x = layer(x)
    #     x = x.view(x.shape[0], -1)
    #     return (x, adv_loss_x)

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
