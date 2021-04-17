import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torchsummary import summary
from network import *
from data import *
from misc import *
from options import *

DATA_PATH = "/raid/veliseev/datasets/dtd/images/"

class fake_args:
    def __init__(self):
        self.config = "/raid/veliseev/dev/psgan_my/configs/config.json"
        self.device = "cuda"
        self.device_ids = [5]
opt = options(fake_args())
# opt = small_opt()


def imshow(img, name=None):
    fig, ax = plt.subplots(frameon=False)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.axis('off')
    img = np.transpose(img.numpy(), (1, 2, 0))
    ax.imshow(img, interpolation='none')

    if name != None:
        fig.tight_layout()
        fig.savefig(name + ".png")
    else:
        fig.show()
    plt.close()

img_size = 256
dataloader = get_loader(
    data_set=get_dtd_data_loader(opt, img_size, batch_size=opt.batch_size),
    batch_size=opt.batch_size,
    shuffle=True,
    # num_workers=self.opt.num_workers
    num_workers=8
)

print(f"DATA lenght {len(dataloader)}")

for i_batch, im in enumerate(dataloader):
    print(im.max(), im.min())
    im = (im+1.0)/2.0
    
    torchvision.utils.save_image(
        im, os.path.join(f"./{i_batch}.png"),
        padding=int(img_size*0.05), normalize=True, nrow=6
    )
    if i_batch == 2:
        break

gen = PSGAN_Generator(opt).cpu()
# Noise
Z_l = torch.rand((opt.batch_size, opt.local_noise_dim, opt.spatial_size, opt.spatial_size), device="cpu") * 2.0 - 1.0
Z_g = torch.rand((opt.batch_size, opt.global_noise_dim, 1, 1), device="cpu") * 2.0 - 1.0
pad = (
    opt.spatial_size // 2 - 1 + opt.spatial_size % 2,
    opt.spatial_size // 2,
    opt.spatial_size // 2 - 1 + opt.spatial_size % 2,
    opt.spatial_size // 2
)
Z_g = F.pad(Z_g, pad, mode='replicate')
# New image
img_size = opt.spatial_size * 2
dataloader = get_loader(
    data_set=get_dtd_data_loader(opt, img_size, batch_size=opt.batch_size),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8
)
loader_it = iter(dataloader)
fixed_noise = [Z_l, Z_g, next(loader_it).cpu()]
print(fixed_noise[-1].shape)
fake = gen(*fixed_noise, -1).detach().cpu()
torchvision.utils.save_image(
    fake, f"./fake.png",
    padding=int(img_size*0.05), normalize=True, nrow=5
)
