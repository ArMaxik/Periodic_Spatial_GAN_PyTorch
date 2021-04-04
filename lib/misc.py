import numpy as np

from tqdm import tqdm

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import re


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
        if new_key == key:
            continue
        state_dict[new_key] = state_dict[key]
        state_dict.pop(key)

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]



def make_video(img_path, save_path, name):
    i = 0
    imgs = []
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    imgs_f = [img_f for img_f in os.listdir(img_path)]
    imgs_f.sort(key=natural_keys)

    _imgs_f = tqdm(imgs_f, ncols=100, desc="video")
    for img_f in _imgs_f:
        img = cv2.imread(os.path.join(img_path, img_f))
        if img is None:
            break
        plt.axis('off')

        imgs.append([plt.imshow(img[:,:,::-1], animated=True)])

        i += 1

    ani = animation.ArtistAnimation(fig, imgs, interval=600, repeat_delay=100, blit=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, bitrate=25000, codec='mpeg4')
    ani.save(os.path.join(save_path, name+'_hist.mp4'), writer=writer)
