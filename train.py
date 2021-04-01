from lib.options import options
from lib.model import PSGAN
from lib.misc import prep_dirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)
    parser.add_argument("-d", "--device", help="device for running", metavar="Device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device_ids", help="device ids", metavar ="IDs", nargs='+', type=int, default=[0])


    args = parser.parse_args()
    opt = options(args)
    opt.show()
    prep_dirs(opt)

    psgan = PSGAN(opt)

    # Z_l, Z_g  = psgan.generate_noise(opt.batch_size)
    # print(Z_l.shape, Z_g.shape)

    psgan.train()

