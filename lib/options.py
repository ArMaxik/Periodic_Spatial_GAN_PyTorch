import json
import torch
import os

class options:
    def __init__(self, args):
        config = args.config
        with open(config, 'r') as f:
            config_j = json.loads(f.read())
        for k, v in config_j.items():
            setattr(self, k, v)
        # Select device
        if args.device == "cuda":
            if not torch.cuda.is_available():
                print("Cuda is not avaliable")
                setattr(self, "device", torch.device("cpu"))
            else:
                setattr(self, "device", torch.device(f"cuda:{args.device_ids[0]}"))
        else:
            setattr(self, "device", torch.device("cpu"))
        # Device ids
        setattr(self, "device_ids", args.device_ids)
        # Number of devices
        setattr(self, "num_devices", len(args.device_ids))
        # Training folder
        work_folder = os.path.join(self.out_folder, self.exp_name)
        setattr(self, "work_folder", work_folder)

    def dump(self, path):
        res = ""
        for k, v in self.__dict__.items():
            res += f"{k:.<20} {v}\n"

        with open(os.path.join(path, "options.txt"), 'w') as f:
            f.write(res)

    def show(self):
        print("--- Training options ---")
        for k, v in self.__dict__.items():
            print(f"{k:.<20} {v}")
