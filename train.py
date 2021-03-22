from lib.options import options
from lib.network import PSGAN_Generator
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="network config", metavar="Config", type=str, required=True)

    args = parser.parse_args()
    opt = options(args.config)
    PSGAN_Generator(opt)

