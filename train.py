from lib.options import options
from lib.network import PSGAN_Generator, PSGAN_Discriminator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
    parser.add_argument("-d", "--device", help="device for running", metavar="Device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device_ids", help="device ids", metavar ="IDs", nargs='+', type=int, default=[0])


    args = parser.parse_args()
    opt = options(args)
    PSGAN_Generator(opt)
    PSGAN_Discriminator(opt)

    opt.show()

