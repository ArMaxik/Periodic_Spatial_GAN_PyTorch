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
        # self.opt.image_list = ["book_page"]
        # self.opt.dataset = "/raid/veliseev/datasets/dtd/images/"

        self.img_size = self.opt.spatial_size*(2**len(self.opt.gen_conv_channels))
        self.dataloader = get_loader(
            data_set=get_dtd_data_loader(self.opt, self.img_size, self.opt.batch_size),
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=8
        )

    def generate(self, spatial_size):
        Z_l, Z_g, imgs = self.generate_noise(self.opt.batch_size, spatial_size)
        img = self.gen(Z_l, Z_g, imgs, spatial_size=spatial_size).detach().cpu()
        return img, imgs

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
        loader_it = iter(self.dataloader)
        imgs = next(loader_it).to(self.opt.device)
        return (Z_l, Z_g, imgs)

class PSGAN():
    def __init__(self, opt):
        self.opt = opt

        self.gen = PSGAN_Generator(self.opt).to(self.opt.device)
        self.dis = PSGAN_Discriminator(self.opt).to(self.opt.device)
        # DataParallel
        self.gen = nn.DataParallel(self.gen, device_ids=self.opt.device_ids)
        self.dis = nn.DataParallel(self.dis, device_ids=self.opt.device_ids)

    def train(self):
        print("Training started")
        self._setup_train()

        # TODO: assert(len(self.opt.gen_conv_channels) == len(self.opt.dis_conv_channels))

        # TODO: rename gen_conv_channels to conv_channels
        for i in range(len(self.opt.gen_conv_channels)-1):
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.opt.lr_g, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.opt.lr_d, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
            self.transition = False
            self.alpha = -1  # No transition
            # tqdm with fixed lenght to prevent resizing glitch
            epochs = tqdm(range(self.opt.epochs), ncols=100, desc="train")
            for epoch in epochs:
                self._train_one_epoch()
                
                self._make_stat(epoch)
                self._save_weights()
                tqdm.write(f"[#{epoch+1}] size {self.img_size} | {self.opt.exp_name} | dloss: {self.d_loss:.5f}, gloss: {self.g_loss:.5f}")

            self._save_progress_image(os.path.join(self.opt.work_folder, f"img_final_{i}.png"))

            self.transition = True
            self.gen.module.add_block()
            self.dis.module.add_block()
            self.gen.module.to(self.opt.device)
            self.dis.module.to(self.opt.device)
            self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.opt.lr_g, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
            self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.opt.lr_d, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))

            self.img_size = self.opt.spatial_size*(2**(i+2))
            self.dataloader = get_loader(
                data_set=get_dtd_data_loader(self.opt, self.img_size, self.opt.batch_size),
                batch_size=self.opt.batch_size,
                shuffle=True,
                num_workers=32
            )
            # Update fixed noise
            # TODO: refactor
            tmp_loader = get_loader(
                data_set=get_dtd_data_loader(self.opt, self.img_size, 36),
                batch_size=36,
                shuffle=True,
                num_workers=32
            )
            tmp_loader = iter(tmp_loader)
            self.fixed_noise[-1] = next(tmp_loader).to(self.opt.device)
            vutils.save_image(
                self.fixed_noise[2], os.path.join(self.opt.work_folder, f"fixed_batch_{self.img_size}.png"),
                padding=int(self.img_size*0.05), normalize=True, nrow=6
            )

            # Aplpha
            alpha_inc = 1.0 / (self.opt.epochs + 1)
            self.alpha = alpha_inc
            # tqdm with fixed lenght to prevent resizing glitch
            epochs = tqdm(range(self.opt.epochs), ncols=100, desc="train")
            for epoch in epochs:
                self._train_one_epoch()
                self.alpha += alpha_inc
                self._make_stat(epoch)
                self._save_weights()
                tqdm.write(f"[#{epoch+1}] size {self.img_size} | {self.opt.exp_name} | dloss: {self.d_loss:.5f}, gloss: {self.g_loss:.5f}")
            self._save_progress_image(os.path.join(self.opt.work_folder, f"img_final_{i}.png"))
            
            self.gen.module.end_transition()
            self.dis.module.end_transition()

        # Final iteration
        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.opt.lr_g, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.opt.lr_d, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
        self.transition = False
        self.alpha = -1  # No transition
        # tqdm with fixed lenght to prevent resizing glitch
        epochs = tqdm(range(self.opt.epochs), ncols=100, desc="train")
        for epoch in epochs:
            self._train_one_epoch()
            
            self._make_stat(epoch)
            self._save_weights()
            tqdm.write(f"[#{epoch+1}] size {self.img_size} | {self.opt.exp_name} | dloss: {self.d_loss:.5f}, gloss: {self.g_loss:.5f}")

        self._save_progress_image(os.path.join(self.opt.work_folder, f"img_final_{i}.png"))


    def _setup_train(self):
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.d_loss = 0.0
        self.g_loss = 0.0

        self.criterion = nn.BCELoss()
        self.MSEloss = nn.MSELoss()
        
        self.img_size = self.opt.spatial_size * 2
        self.dataloader = get_loader(
            data_set=get_dtd_data_loader(self.opt, self.img_size, self.opt.batch_size),
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=32
        )
        # Save fixed batch
        tmp_loader = get_loader(
            data_set=get_dtd_data_loader(self.opt, self.img_size, 36),
            batch_size=36,
            shuffle=True,
            num_workers=32
        )
        tmp_loader = iter(tmp_loader)
        # self.fixed_noise = self.generate_noise(36)
        self.fixed_noise = [*self.generate_noise(36), next(tmp_loader).to(self.opt.device)]
        vutils.save_image(
            self.fixed_noise[2], os.path.join(self.opt.work_folder, f"fixed_batch_{self.img_size}.png"),
            padding=int(self.img_size*0.05), normalize=True, nrow=6
        )

    def _save_progress_image(self, path):
        with torch.no_grad():
            fake = self.gen(*self.fixed_noise, self.alpha).detach().cpu()
            vutils.save_image(
                fake, path,
                padding=int(self.img_size*0.05), normalize=True, nrow=6
            )

    def _make_stat(self, epoch):
        # Save progress image
        if epoch % 10 == 0:
            self._save_progress_image(os.path.join(self.opt.work_folder + f"/progress/img_{len(self.G_losses)}.png"))

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
    # TODO: user Tensor.repeat()
    def generate_noise(self, batch_size):
        # TODO: Z_l seems like not right thing
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

    def _gradien_penalty(self, imgs_real, imgs_fake):
        b, c, h, w = imgs_real.shape
        epsilon = torch.rand((b, 1, 1, 1), device=self.opt.device).repeat(1, c, h, w)
        interpolate = epsilon*imgs_real + (1.0 - epsilon)*imgs_fake
        interpolate.requires_grad_(True)

        d_interpolate = self.dis(interpolate, self.alpha).view(-1)

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
        tqdm.write(f"{penalty.min()} {penalty.max()}")
        return self.opt.lambda_coff * penalty

    def _train_discriminator(self):
        self.op_dis.zero_grad()
        ### Train with real images
        d_real_out = self.dis(self.data_device, self.alpha)[0]

        ### Train with fake images
        # Generate fake images
        Z_l, Z_g = self.generate_noise(self.current_batch)
        imgs_fake = self.gen(Z_l, Z_g, self.data_device, self.alpha)
        # Calculate gradient
        d_fake_out = self.dis(imgs_fake, self.alpha)[0]
        
        # tqdm.write(f"{d_fake_out.mean()} + {d_real_out.mean()} + {self._gradien_penalty(self.data_device, imgs_fake)}")

        gp = self._gradien_penalty(self.data_device, imgs_fake)
        d_real_out = d_real_out.view(-1).mean()
        d_fake_out = d_fake_out.view(-1).mean()
        self.d_loss = d_fake_out - d_real_out + gp
        self.d_loss += self.opt.eps_drift * torch.mean(d_real_out ** 2)
        self.d_loss.backward()
        # Optimize weights
        self.op_dis.step()

    def _train_generator(self):
        self.op_gen.zero_grad()
        # Generate fake images
        Z_l, Z_g = self.generate_noise(self.current_batch)
        imgs_fake = self.gen(Z_l, Z_g, self.data_device, self.alpha)
        # Calculate gradient
        g_fake_out = self.dis(imgs_fake, self.alpha)
        # _, prelast_layer_real = self.dis(self.data_device, self.alpha)
        
        self.g_loss = -g_fake_out.view(-1).mean()
        # self.g_loss += self.MSEloss(imgs_fake, self.data_device) * self.opt.MSE_coff
        # self.g_loss += self.MSEloss(prelast_layer_fake, prelast_layer_real) * self.opt.adv_coff
        self.g_loss.backward()
        # Optimize weights
        self.op_gen.step()

    def _save_weights(self):
        g_w = self.gen.state_dict()
        d_w = self.dis.state_dict()
        remove_module_from_state_dict(g_w)
        remove_module_from_state_dict(d_w)

        torch.save(g_w, os.path.join(self.opt.work_folder, 'c_gen.pth'))
        torch.save(d_w, os.path.join(self.opt.work_folder, 'c_dis.pth'))