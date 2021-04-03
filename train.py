from lib.options import options
from lib.model import PSGAN
from lib.misc import prep_dirs, make_video

import os
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

    psgan.train()

    make_video(os.path.join(opt.work_folder, "progress"), opt.work_folder, opt.exp_name)


