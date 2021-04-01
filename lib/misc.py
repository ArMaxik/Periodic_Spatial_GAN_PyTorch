import os

def prep_dirs(opt):
    if not os.path.isdir(opt.work_folder):
        print(f"{opt.work_folder} is not exsists! Creating...")
        os.makedirs(opt.work_folder)

    progress_dir = os.path.join(opt.work_folder, "progress")
    if not os.path.isdir(progress_dir):
        print(f"{progress_dir} is not exsists! Creating...")
        os.makedirs(progress_dir)

def remove_module_from_state_dict(state_dict):
    old_keys = list(state_dict.keys())
    for key in old_keys:
        new_key = key.replace('module.', '')
        state_dict[new_key] = state_dict[key]
        state_dict.pop(key)