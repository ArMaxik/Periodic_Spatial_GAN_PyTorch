import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from lib.network import PSGAN_Generator, PSGAN_Discriminator
from lib.data import get_dtd_train_loader

class PSGAN():
    def __init__(self, opt):
        self.opt = opt

        self.gen = PSGAN_Generator(self.opt).to(self.opt.device)
        self.dis = PSGAN_Generator(self.opt).to(self.opt.device)
        # TODO: add DataParallel


    def train(self):
        print("Training started")
        # tqdm with fixed lenght to prevent resizing glitch
        # self.pbar = tqdm(bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
        epochs = tqdm(range(args.epochs), ncols=100, desc="train")

        for epoch in epochs:
            self.train_one_epoch()

    def setup_train(self):
        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        self.criterion = nn.BCELoss()
        
        op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.opt.lr_g, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))
        op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.opt.lr_d, weight_decay=1e-8, betas=(self.opt.b1, self.opt.b2))

        self.real_label = torch.full((self.opt.batch_size, self.opt.spatial_size*self.opt.spatial_size), 1.0)
        self.fake_label = torch.full((self.opt.batch_size, self.opt.spatial_size*self.opt.spatial_size), 0.0)

        self.fixed_noise = self.generate_noise(self.opt.batch_size)

        self.train_loader = get_loader(
                                data_set=get_dtd_train_loader(args, img_size),
                                batch_size=self.opt.batch_size,
                                shuffle=True,
                                # num_workers=self.opt.num_workers
                                num_workers=8
                            )

    def generate_noise(self, batch_size):
        # TODO: Z_l seems like not right thing
        Z_l = torch.rand((batch_size, self.opt.local_noise_dim, self.opt.spatial_size, self.opt.spatial_size), device=self.opt.device)
        Z_g = torch.rand((batch_size, self.opt.global_noise_dim, 1, 1), device=self.opt.device)
        Z_g = F.pad(Z_g, (0, 0, self.opt.spatial_size, self.opt.spatial_size), mode='replicate')

        return (Z_g, Z_l)

    def train_one_epoch(self):
        for i, data in enumerate(tqdm(self.dataloader), 0):
            self.data_device = data.to(self.device)
            
            mini_it = i % (self.opt.g_it + self.opt.d_it-1)
            # Teach discriminator for d_it iterations
            if mini_it < self.opt.d_it:
                self.train_discriminator()
            # Teach generator for g_it iterations
            if mini_it >= self.opt.d_it-1:
                self.train_generator()

    def train_discriminator(self):
        self.op_dis.zero_grad()
        # Real image
