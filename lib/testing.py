import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torchsummary import summary
from network import *
from data import *
from misc import *

DATA_PATH = "/raid/veliseev/datasets/dtd/images/"
class small_opt:
    def __init__(self):
        self.dataset = "/raid/veliseev/datasets/dtd/images/"
        self.image_list = "/raid/veliseev/dev/psgan_my/train_names.txt"
opt = small_opt()


def imshow(img, name=None):
    fig, ax = plt.subplots()
    img = np.transpose(img.numpy(), (1, 2, 0))
    ax.imshow(img, interpolation='none')
    ax.axis('off')

    if name != None:
        fig.tight_layout()
        fig.savefig(name + ".png")
    else:
        fig.show()
    plt.close()

img_size = 256
dataloader = get_loader(
    data_set=get_dtd_data_loader(opt, img_size),
    batch_size=16,
    shuffle=True,
    # num_workers=self.opt.num_workers
    num_workers=8
)

print(f"DATA lenght {len(dataloader)}")

img_list = []
for i_batch, im in enumerate(dataloader):
    print(im.max(), im.min())
    im = (im+1.0)/2.0
    
    imshow(torchvision.utils.make_grid(im, nrow=4), name=str(i_batch))
    img_n = torchvision.utils.make_grid(im, nrow=4).numpy()
    img_list.append(img_n)
    if i_batch == 2:
        break
print(img_n.shape, img_n.dtype, np.min(img_n), np.max(img_n))
fig = plt.figure(figsize=(12,12))

