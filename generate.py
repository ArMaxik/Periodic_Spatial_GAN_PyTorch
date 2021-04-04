from lib.model import Texture_generator
from lib.options import options

import torchvision.utils as vutils
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output directory", metavar="Output", type=str, required=True)
parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
parser.add_argument("-n", "--number", help="number of generated images", metavar ="Number", type=int, default=1)
parser.add_argument("-d", "--device", help="using device", metavar ="Device", type=str, choices=["cpu", "cuda"], default="cpu")
parser.add_argument("-s", "--spatial", help="spatial size", metavar ="Size", type=int, default=6)
parser.add_argument("--device_ids", help="device ids", metavar ="IDs", nargs='+', type=int, default=[0])


args = parser.parse_args()

with open(args.config, 'r') as config:
    config_j = json.loads(config.read())

args = parser.parse_args()
opt = options(args)
opt.spatial_size = args.spatial

gen = Texture_generator(opt)
for i in range(args.number):
    print(f"Generating image: {i+1:{len(str(args.number))}}/{args.number}")
    img = gen.generate(opt.spatial_size)
    vutils.save_image(
            img,
            os.path.join(args.output + f"{i}.png"),
            padding=0,
            nrow=1,
            normalize=True,
        )