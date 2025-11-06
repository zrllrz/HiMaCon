import os, sys, shutil, re
import cv2
sys.path.append(os.path.abspath('../src'))
from datetime import datetime
import PIL
from PIL import Image, ImageDraw, ImageFont
font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
font = ImageFont.truetype(font_path, 32)

import matplotlib.pyplot as plt
import argparse
import yaml, json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

import pytorch_lightning as pl

from data import MMDemos_small
from modules.VQ import VQConfig, SinkhormKnoppConfig, EMAConfig, TrainTimeMemoConfig
from modules.GPT import MNConfig
from modules.coe_adjust import CoefficientsAdjuster

try:
    from hminfocon import HMInfoCon
    print("Import HMInfoCon successful")
except ImportError:
    print("Import HMInfoCon failed")
# from hminfocon import hMInfoCon

from lr_scheduler import CosineAnnealingLRWarmup
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import MySaveLogger
from util import safe_handler, plot_functions, copy_file

from diffusers import StableDiffusionPipeline
import torchvision.transforms as T


# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
# # You only need the VAE encoder part for encoding the image
# vae_part = pipe.vae


def tensor_to_pil_image(decoded_image_tensor):
    """
    Converts a decoded image tensor to a PIL Image.
    
    Args:
    - decoded_image_tensor (torch.Tensor): Decoded image tensor from the VAE decoder.
    
    Returns:
    - PIL.Image: The decoded image in PIL format.
    """
    # Detach tensor, move to CPU, and denormalize (scale back to [0, 1] range)
    decoded_image_tensor = decoded_image_tensor.squeeze(0).cpu()
    decoded_image_tensor = (decoded_image_tensor * 0.5) + 0.5  # Reverse normalization [-1, 1] to [0, 1]
    decoded_image = T.ToPILImage()(decoded_image_tensor)
    return decoded_image


@torch.no_grad()
def label_handler(hminfocon_model, in_list_total, len_seg, disable_segment=False, vae_part=None):
    assert isinstance(hminfocon_model, HMInfoCon),f"Please use HMInfoCon Model, Now you give me model type {hminfocon_model}"
    in_list_total = [torch.tensor(m, dtype=torch.float32, device=hminfocon_model.device).unsqueeze(0) for m in in_list_total]
    total_len = in_list_total[0].shape[1]
    
    # Start Labeling
    token_pos = 0  # record next token pos to label
    context_size = 1    # current context size:
    labels = []
    
    while token_pos < total_len:
        in_list = [m[:, token_pos-context_size+1:token_pos+1, :] for m in in_list_total]
        # prepare into input
        timesteps = torch.tensor([[token_pos - context_size + 1]], device=hminfocon_model.device)
        
        # encoding
        z_soft = hminfocon_model.label_single(in_list, timesteps)
        z_soft = z_soft.squeeze(0)[-1]
        
        # save encoding
        labels.append(z_soft)
        
        # updata token pos and context_size
        token_pos += 1
        context_size = min(len_seg, context_size + 1)
    
    labels = torch.stack(labels)
    if disable_segment:
        return labels, None, None
    
    cos_sim = F.cosine_similarity(labels.unsqueeze(0), labels.unsqueeze(1), dim=-1)
    cos_sim = torch.clip(cos_sim, min=-1.0, max=1.0)
    arccos_dis = torch.div(torch.arccos(cos_sim), 3.15)
    
    # use bisection method to get all possible segmentation
    possible_seg, total = [], 0
    collect_eps = []
    bisec_queue = []
    bisec_map = {}
    
    _, possible_seg_0, _ = hminfocon_model.get_goal_state_tolerance(in_list_total, arccos_dis.unsqueeze(0), 0.0)
    _, possible_seg_last, _ = \
        hminfocon_model.get_goal_state_tolerance(in_list_total,
                                                 arccos_dis.unsqueeze(0),
                                                 (hminfocon_model.N_th-1)/hminfocon_model.N_th)
        
    if torch.all(possible_seg_0 == possible_seg_last):
        possible_seg.append(possible_seg_0)
        collect_eps.append(1.0)
        
        total += 1
        # no other possible now...
    else:
        # start bisection discovery
        possible_seg.append(possible_seg_0)
        collect_eps.append(0.0)
        
        possible_seg.append(possible_seg_last)
        collect_eps.append(1.0)
        
        bisec_map[0] = 0
        bisec_map[hminfocon_model.N_th-1] = 1
        bisec_queue.append([0, hminfocon_model.N_th-1])
        total = 2
    
        while bisec_queue:
            be = bisec_queue.pop(0)
            if be[1] - be[0] <= 1:
                continue
            e_mid = be[0] + (be[1] - be[0]) // 2
            assert be[0] < e_mid < be[1]
            epsilon = e_mid / hminfocon_model.N_th
            _, future_state_indices, _ = hminfocon_model.get_goal_state_tolerance(in_list_total, arccos_dis.unsqueeze(0), epsilon)
            further_bm = torch.any(possible_seg[bisec_map[be[0]]] != future_state_indices)
            further_me = torch.any(possible_seg[bisec_map[be[1]]] != future_state_indices)
            
            if further_bm and further_me:
                possible_seg.append(future_state_indices)
                collect_eps.append(epsilon)
                
                bisec_map[e_mid] = total
                total += 1
                bisec_queue.append([be[0], e_mid])
                bisec_queue.append([e_mid, be[1]])
            
            elif further_bm and (not further_me):
                bisec_map[e_mid] = bisec_map[be[1]]
                bisec_queue.append([be[0], e_mid])
                
            elif (not further_bm) and further_me:
                bisec_map[e_mid] = bisec_map[be[0]]
                bisec_queue.append([e_mid, be[1]])
            else:
                assert (not further_bm) and (not further_me)
    
    print(f'using bisection, we discover {len(possible_seg)} segmentation level')
    return labels, possible_seg, collect_eps
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_config', type=str, help="path of config file")
    parser.add_argument('--change_set', type=str, default='',
                        help="if not an empty string, will switch to a new set for ZERO SHOT LABELING")
    parser.add_argument('--disable_segment', action="store_true", help="do not save hierarchical segmentations")
    parser.add_argument('--ckpt', type=int, help="checkpoint number", default=-1)
    args = parser.parse_args() # args only have a path
    print(args)
    
    # Load the hyperparameters from the path
    with open(args.label_config, "r") as f:
        label_config = yaml.safe_load(f)
    print(label_config)
    
    # Get necessary information to reconstruct data and models
    data_config = label_config['data']
    model_config = label_config['model']
    print('### DATA CONFIG ###\n', data_config)
    print('\n### MODEL CONFIG ##\n', model_config)
    
    # prepare dataset
    print(f"\nPreparation of Dataset Begin :(")
    if args.change_set != '':
        print(f'Change to this set for zero-shot labeling: {args.change_set}')
        data_config_path = os.path.join(os.path.dirname(data_config['path']), args.change_set)
    else:
        print(f'Use original set')
        data_config_path = data_config['path']
    print(f'data path:\n{data_config_path}')
    
    evalset = MMDemos_small(
        path=data_config_path,
        handler=data_config['handler'],
        train_split=data_config['train_split'],
        multiplier=1,
        seed=data_config['seed'],
        seg_len=data_config['seg_len'],
        init_mode='eval',
    )
    print(f"Preparation of Dataset End :)\n")
    
    # Let's check if we load data right...
    modal0 = evalset.DataList[0]
    print(len(modal0))
    ## see an example
    print(modal0[0])
    
    # prepare model
    print(f"\nInitial Model :(")
    block_size = data_config['seg_len']
    block_size = block_size if block_size is not None else evalset.max_traj_len
    block_scale = 1
    
    print(f"\nInitial Encoder :(")
    enc_config = MNConfig(n_embd=model_config['n_embd'],
                          n_head=model_config['n_head'],
                          attn_pdrop=model_config['dropout'],
                          resid_pdrop=model_config['dropout'],
                          embd_pdrop=model_config['dropout'],
                          block_size=block_size,
                          block_scale=block_scale,
                          use_causal=model_config['encoder']['use_causal'],
                          n_layer=model_config['encoder']['n_layer'],
                          max_timestep=evalset.max_traj_len,
                          hyper_dim=model_config['encoder']['hyper_dim'])
    print(f"Initial Encoder End :)\n")
    
    print(f"\nInitial Reconstructor :(")
    rec_config = MNConfig(n_embd=model_config['n_embd'],
                          n_head=model_config['n_head'],
                          attn_pdrop=model_config['dropout'],
                          resid_pdrop=model_config['dropout'],
                          embd_pdrop=model_config['dropout'],
                          block_size=block_size,
                          block_scale=block_scale,
                          use_causal=model_config['reconstructor']['use_causal'],
                          n_layer=model_config['reconstructor']['n_layer'],
                          max_timestep=evalset.max_traj_len,
                          hyper_dim=model_config['reconstructor']['hyper_dim'])
    print(f"Initial Reconstructor End :)\n")
    
    print(f"\nInitial Goal-Predictor :(")
    goal_config = MNConfig(n_embd=model_config['n_embd'],
                           n_head=model_config['n_head'],
                           attn_pdrop=model_config['dropout'],
                           resid_pdrop=model_config['dropout'],
                           embd_pdrop=model_config['dropout'],
                           block_size=block_size,
                           block_scale=block_scale,
                           use_causal=model_config['goal']['use_causal'],
                           n_layer=model_config['goal']['n_layer'],
                           max_timestep=evalset.max_traj_len,
                           hyper_dim=model_config['goal']['hyper_dim'])
    print(f"Initial Goal-Predictor End :)\n")
    
    # Create model
    print(args.label_config)
    model_name = args.label_config.split('/')[-2]
    print(f"model_name: {model_name}")
    hminfocon_model = HMInfoCon(enc_config=enc_config,
                                rec_config=rec_config,
                                goal_config=goal_config,
                                th_config=model_config['th_config'],
                                opt_config=None,
                                sch_config=None,
                                modal_dims=evalset.modal_dims,
                                modal_usage=model_config['modal_usage'],
                                modes=data_config['loss_modes'],
                                name=model_name,
                                metric_path='do_not_use')
    
    # load from checkpoints
    ckpt_dir = os.path.join(os.path.dirname(args.label_config), 'ckpts')
    print("ckpt path:", ckpt_dir)
    ### read a select
    ckpts = os.listdir(ckpt_dir)
    ### select out .pth
    ckpts = sorted([int(file[:-4]) for file in ckpts if (file.endswith('.pth') and file[:5] != 'final')])
    print(ckpts)
    print(f"Please select from {min(ckpts)} ~ += {label_config['train']['save_ckpt_every']} ~ {max(ckpts)} (Enter to select the last ckpt)")
    if args.ckpt != -1:
        ckpt_id = int(args.ckpt)
    else:
        ckpt_selection = input()
        if len(ckpt_selection) == 0:
            ckpt_id = max(ckpts)
        else:
            ckpt_id = int(ckpt_selection)
    ### load!!!
    ckpt = torch.load(f"{ckpt_dir}/{ckpt_id}.pth", map_location=torch.device('cpu'))
    state_dict_from_ckpt = ckpt['module']
    hminfocon_model = hminfocon_model.cuda()
    hminfocon_model.load_state_dict(state_dict_from_ckpt, strict=False)
    hminfocon_model = hminfocon_model.eval()
    print(f"[{model_name}]: Load pretrained parameter from {ckpt_dir}/{ckpt_id}.pth")
    
    # segs' dir
    parent_dir = os.path.dirname(args.label_config)[3:]
    segs_dir = os.path.join(parent_dir, f'labels-{ckpt_id}')
    segs_dir = "../" + segs_dir
    print(f"{segs_dir}")
    # input()
    if args.change_set != '':
        segs_dir += '-' + args.change_set
    print(f'save labels at:\n{segs_dir}')
    os.makedirs(segs_dir, exist_ok=True)
    
    ### label!!!
    idx2path_map = {}
    path2idx_map = {}
    for traj_idx in evalset.train_idx:
        # if traj_idx not in [1, 701, 802, 2678, 2889, 2993, 3292, 3342]:
        #     continue
        print(traj_idx)
        
        data = [modal[traj_idx] for modal in evalset.DataList]
        in_list_total = [m['data'] for m in data]
        print(data[0].keys())
        data_path = data[0]['data_path']
        print(f'traj_idx:\t{traj_idx}')
        print(f'data_path:\t{data_path}')
        
        idx2path_map[f'traj_{traj_idx}'] = data_path
        path2idx_map[data_path] = f'traj_{traj_idx}'
        
        # here {labels} are embedding [T, d],
        # and {possible_seg} is a set of segmentation based on threshold, recording goal state
        labels, possible_seg, collect_eps = label_handler(hminfocon_model, in_list_total, data_config['seg_len'], args.disable_segment)
        if possible_seg is not None:
            possible_seg_tensor = torch.cat(possible_seg, dim=0)
            collect_eps_tensor = torch.tensor(collect_eps)
        
        print(labels.shape)
        if possible_seg is not None:
            print(possible_seg_tensor.shape)
            print(collect_eps_tensor.shape)
            for eps_here in collect_eps_tensor:
                print(f"{float(eps_here):.3f}", end=' ')
            print()
        
        # save as numpy
        labels_npy = labels.detach().cpu().numpy()
        if possible_seg is not None:
            possible_seg_npy = possible_seg_tensor.detach().cpu().numpy()
            collect_eps_npy = collect_eps_tensor.detach().cpu().numpy()
        save_path = f'{segs_dir}/traj_{traj_idx}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f'{save_path}/emb.npy', labels_npy)
        if possible_seg is not None:
            np.save(f'{save_path}/possible_seg.npy', possible_seg_npy)
            np.save(f'{save_path}/collect_eps.npy', collect_eps_npy)

    ### dump json for idx/path mapping
    with open(f'{segs_dir}/idx2path.json', 'w') as jf1:
        json.dump(idx2path_map, jf1, indent=4)
    with open(f'{segs_dir}/path2idx.json', 'w') as jf2:
        json.dump(path2idx_map, jf2, indent=4)
        
        
