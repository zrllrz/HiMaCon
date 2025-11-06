import os, sys, shutil, random
import cv2
from PIL import Image
sys.path.append(os.path.abspath('../minfoncon/src'))
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import yaml, json
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

import pytorch_lightning as pl

# from data import MMDemos_small
# from modules.VQ import VQConfig, SinkhormKnoppConfig, EMAConfig, TrainTimeMemoConfig
# from modules.GPT import MNConfig
# from modules.coe_adjust import CoefficientsAdjuster

# try:
#     from hminfocon import HMInfoCon
#     print("Import HMInfoCon successful")
# except ImportError:
#     print("Import HMInfoCon failed")
# from hminfocon import hMInfoCon

# from lr_scheduler import CosineAnnealingLRWarmup
# from pytorch_lightning.callbacks import ModelCheckpoint
# from callbacks import MySaveLogger
from util import safe_handler, plot_functions, copy_file
# from label_selections import RatioLargestConfig, ThresholdConfig, IntegrationConfig, labelSelection


@torch.no_grad()
def get_h_segments(possible_seg: torch.Tensor):
    """generate [[t_1, t_2-1],[t_2, t_3-1],...] to indicate possible segmentation

    Args:
        possible_seg (torch.Tensor): [B, T] labeling the GOAL TIME_STEP
    """
    B, T = possible_seg.shape
    h_segments = []
    for b in range(B):
        seg_goal = torch.cat([possible_seg[b],
                              torch.tensor([-1], device=possible_seg.device, dtype=possible_seg.dtype)],
                             dim=0)
        assert seg_goal.shape[0] == T+1
        
        h_seg = []
        t_begin = 0
        for t in range(1, T+1, 1):
            if seg_goal[t] != seg_goal[t_begin]:
                h_seg.append([t_begin, t-1])
                t_begin = t
        # print(f'h_seg:', end='\t')
        # for interval in h_seg:
        #     print(f'[{interval[0]}, {interval[1]}]', end=' ')
        # print(seg_goal)
        h_segments.append(h_seg)
    print(f'h_segments have {len(h_segments)} levels')
    return h_segments
    

def img_seq_to_gif(img_seq: np.ndarray, output_path: str, duration: int = 100):
    imgs = [Image.fromarray(img_seq[t]) for t in range(img_seq.shape[0])]
    imgs[0].save(
        output_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0
    )   



def main_scripted_raw():
    path = '/media/disk4/lrz/MInfoCon/minfocon/logs/stage-1/stage-1_2024-10-19_20-43-56/labels-120000'
    # get mapping
    with open(f'{path}/idx2path.json') as f:
        idx2path = json.load(f)
    traj_list = os.listdir(path)
    traj_paths = [path for path in traj_list if path.startswith('traj')]
    traj_paths_random = random.sample(traj_paths, 10)
    
    samples_dir = f'samples-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(samples_dir, exist_ok=False)
    with open(f'{samples_dir}/log.txt', 'w') as flog:
        flog.write(f'from labels at:\n{path}')
    
    for i_sample, traj_idx in enumerate(traj_paths_random):
        os.makedirs(f'{samples_dir}/sample-{i_sample}', exist_ok=False)
        traj_path = idx2path[traj_idx]
        print(traj_idx, ':\t', traj_path)
        img_list = os.listdir(traj_path)
        img_paths = [ipath for ipath in img_list if ipath.endswith('.jpg')]
        img_T = len(img_paths)
        imgs = []
        for t in range(img_T):
            imgs.append(Image.open(f'{traj_path}/im_{t}.jpg'))
            
        possible_seg = np.load(f'{path}/{traj_idx}/possible_seg.npy')
        # print(possible_seg.shape)
        possible_seg = torch.tensor(possible_seg).cuda()
        h_segments = get_h_segments(possible_seg)
        # sort with number of 'steps'
        len_h_segments = torch.tensor([len(h_segment) for h_segment in h_segments])
        sorted_len_h_segments = torch.sort(len_h_segments)
        
        for i, i_h in enumerate(sorted_len_h_segments.indices):
            # print(i_h)
            os.makedirs(f'{samples_dir}/sample-{i_sample}/{i:02}-len_{len(h_segments[i_h])}')
            for i_seg, h_inter in enumerate(h_segments[i_h]):
                img_seg = imgs[h_inter[0]:h_inter[1]+1]
                # print(len(img_seg))
                for frame in img_seg:
                    print(frame)
                img_seg[0].save(
                    f'{samples_dir}/sample-{i_sample}/{i:02}-len_{len(h_segments[i_h])}/step-{i_seg}.gif', save_all=True, append_images=img_seg[1:], duration=500, loop=0
                )
                
        # input()
            
        
        # imgs[0].save(
        #     f'{samples_dir}/{}', save_all=True, append_images=imgs[1:], duration=500, loop=0
        # )
        # input()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main_scripted_raw()
    exit(0)
