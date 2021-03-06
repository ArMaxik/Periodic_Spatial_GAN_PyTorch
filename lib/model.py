import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import os

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

from lib.network import PSGAN_Generator, PSGAN_Discriminator
from lib.data import get_dtd_data_loader, get_loader
from lib.misc import remove_module_from_state_dict

class Texture_generator():
    def __init__(self, opt):
        self.opt = opt
        self.gen = PSGAN_Generator(self.opt)
        
        print("Loading weights")
        weights = torch.load(self.opt.weights, map_location=self.opt.device)
        print(len(weights))
        self.gen.load_state_dict(weights)
        self.gen.to(self.opt.device)

    def generate(self, spatial_size):
        Z_l, Z_g = self.generate_noise(self.opt.batch_size, spatial_size)
        img = self.gen(Z_l, Z_g).detach().cpu()
        return img

    def generate_noise(self, batch_size, spatial_size):
        Z_l = torch.rand((batch_size, self.opt.local_noise_dim, spatial_size, spatial_size), device=self.opt.device) * 2.0 - 1.0
        Z_g = torch.rand((batch_size, self.opt.global_noise_dim, 1, 1), device=self.opt.device) * 2.0 - 1.0
        pad = (
            spatial_size // 2 - 1 + spatial_size % 2,
            spatial_size // 2,
            spatial_size // 2 - 1 + spatial_size % 2,
            spatial_size // 2
        )
        Z_g = F.pad(Z_g, pad, mode='replicate')

        return (Z_l, Z_g)

class PSGAN():
    def __init__(self, opt):
        self.opt = opt

        self.gen = PSGAN_Generator(self.opt).to(self.opt.device)
        self.dis = PSGAN_Discriminator(self.opt).to(self.opt.device)
        # DataParallel
        if self.opt.num_devices > 1:
            self.gen = nn.DataParallel(self.gen, device_ids=self.opt.device_ids)
            self.dis = nn.DataParallel(self.dis, device_ids=self.opt.device_ids)


    def train(self):
        print("Training started")
        self._setup_train()

        # tqdm with fixed lenght to prevent resizing glitch
        epochs = tqdm(range(self.opt.epochs), ncols=100, desc="train")

        for epoch in epochs:
            self._train_one_epoch()
            
            self._make_stat(epoch)
            self._save_weights()
            tqdm.write(f"[#{epoch+1}] train epoch | {self.opt.exp_name} | dloss: {self.d_loss:.5f}, gloss: {self.g_loss:.5f}")

    def _setup_train(self):
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.d_loss = 0.0
        self.g_loss = 0.0

        self.criterion = nn.BCELoss()
        
        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.opt.lr_g, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.opt.lr_d, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))

        self.fixed_noise = self.generate_noise(36)

        img_size = self.opt.spatial_size*(2**len(self.opt.gen_conv_channels))
        self.dataloader = get_loader(
            data_set=get_dtd_data_loader(self.opt, img_size),
            batch_size=self.opt.batch_size,
            shuffle=True,
            # num_workers=self.opt.num_workers
            num_workers=8
        )

    def _make_stat(self, epoch):
        # Save progress image
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = self.gen(*self.fixed_noise).detach().cpu()
                vutils.save_image(
                    fake, os.path.join(self.opt.work_folder + f"/progress/img_{len(self.G_losses)}.png"),
                    padding=0, normalize=True, nrow=6
                )

        # Draw chart
        self.G_losses.append(self.g_loss.item())
        self.D_losses.append(self.d_loss.item())

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.opt.work_folder, "losses.png"))
        plt.close()

    def generate_noise(self, batch_size):
        Z_l = torch.rand((batch_size, self.opt.local_noise_dim, self.opt.spatial_size, self.opt.spatial_size), device=self.opt.device) * 2.0 - 1.0
        Z_g = torch.rand((batch_size, self.opt.global_noise_dim, 1, 1), device=self.opt.device) * 2.0 - 1.0
        pad = (
            self.opt.spatial_size // 2 - 1 + self.opt.spatial_size % 2,
            self.opt.spatial_size // 2,
            self.opt.spatial_size // 2 - 1 + self.opt.spatial_size % 2,
            self.opt.spatial_size // 2
        )
        Z_g = F.pad(Z_g, pad, mode='replicate')

        return (Z_l, Z_g)

    def _train_one_epoch(self):
        for i, data in enumerate(self.dataloader, 0):
            self.data_device = data.to(self.opt.device)
            self.current_batch = data.shape[0]
            self.real_label = torch.full((self.current_batch, self.opt.spatial_size*self.opt.spatial_size), 1.0, device=self.opt.device)
            self.fake_label = torch.full((self.current_batch, self.opt.spatial_size*self.opt.spatial_size), 0.0, device=self.opt.device)
            
            mini_it = i % (self.opt.g_it + self.opt.d_it-1)
            # Teach discriminator for d_it iterations
            if mini_it < self.opt.d_it:
                self._train_discriminator()
            # Teach generator for g_it iterations
            if mini_it >= self.opt.d_it-1:
                self._train_generator()

    def _train_discriminator(self):
        self.op_dis.zero_grad()
        ### Train with real images
        d_real_out = self.dis(self.data_device)

        ### Train with fake images
        # Generate fake images
        Z_l, Z_g = self.generate_noise(self.current_batch)
        imgs_fake = self.gen(Z_l, Z_g)
        # Calculate gradient
        d_fake_out = self.dis(imgs_fake)
        # Apply cost function
        if self.opt.loss == "WGAN":
            gp = self._gradien_penalty(self.data_device, imgs_fake)
            d_real_out = d_real_out.view(-1).mean()
            d_fake_out = d_fake_out.view(-1).mean()

            self.d_loss = d_fake_out - d_real_out + gp
            self.d_loss += self.opt.eps_drift * torch.mean(d_real_out ** 2)
        else:
            # Otherwise using BCELoss by default
            d_real_loss = self.criterion(d_real_out, self.real_label)
            d_fake_loss = self.criterion(d_fake_out, self.fake_label)

            self.d_loss = (d_fake_loss + d_real_loss) / 2
            self.d_loss /= self.opt.spatial_size * self.opt.spatial_size

        self.d_loss.backward()
        # Optimize weights
        self.op_dis.step()

    def _train_generator(self):
        self.op_gen.zero_grad()
        # Generate fake images
        Z_l, Z_g = self.generate_noise(self.current_batch)
        imgs_fake = self.gen(Z_l, Z_g)
        # Calculate gradient
        g_fake_out = self.dis(imgs_fake)
        # Apply cost function
        if self.opt.loss == "WGAN":
            self.g_loss = -g_fake_out.view(-1).mean()
        else:
            # Otherwise using BCELoss by default
            self.g_loss = self.criterion(g_fake_out, self.real_label)
            self.g_loss /= self.opt.spatial_size * self.opt.spatial_size

        self.g_loss.backward()
        # Optimize weights
        self.op_gen.step()

    def _gradien_penalty(self, imgs_real, imgs_fake):
        b, c, h, w = imgs_real.shape
        epsilon = torch.rand((b, 1, 1, 1), device=self.opt.device).repeat(1, c, h, w)
        interpolate = epsilon*imgs_real + (1.0 - epsilon)*imgs_fake
        interpolate.requires_grad_(True)

        d_interpolate = self.dis(interpolate).view(-1)

        gradients = torch.autograd.grad(
            outputs=d_interpolate,
            inputs=interpolate,
            grad_outputs=torch.ones(d_interpolate.shape, device=self.opt.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(b, -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.opt.lambda_coff * penalty

    def _save_weights(self):
        g_w = self.gen.state_dict()
        d_w = self.dis.state_dict()
        remove_module_from_state_dict(g_w)
        remove_module_from_state_dict(d_w)

        torch.save(g_w, os.path.join(self.opt.work_folder, 'c_gen.pth'))
        torch.save(d_w, os.path.join(self.opt.work_folder, 'c_dis.pth'))