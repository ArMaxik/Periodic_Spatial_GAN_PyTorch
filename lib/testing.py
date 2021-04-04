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
        self.image_list = ["bricks"]
        self.batch_size = 16
opt = small_opt()


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
    data_set=get_dtd_data_loader(opt, img_size),
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
        padding=int(img_size*0.05), normalize=True, nrow=4
    )
    if i_batch == 2:
        break

