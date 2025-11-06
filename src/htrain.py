import os, sys, shutil
sys.path.append(os.path.abspath('../src'))
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl

from data import MMDemos_small
from modules.VQ import VQConfig, SinkhormKnoppConfig, KMeansConfig, EMAConfig, TrainTimeMemoConfig
from modules.GPT import MNConfig
from modules.coe_adjust import CoefficientsAdjuster

try:
    from hminfocon import HMInfoCon
    print("Import HMInfoCon successful")
except ImportError:
    print("Import HMInfoCon failed")

from lr_scheduler import CosineAnnealingLRWarmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from callbacks import MySaveLogger
from util import safe_handler, plot_functions, copy_file


if __name__ == "__main__":
    # load .yaml config file from ../train-configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, help="path of config file")
    args = parser.parse_args()
    print(args)
    
    # Load the hyperparameters from the path
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)
    print(train_config)
    
    # create a log file to save everything during training
    ## name is based on system time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_name = os.path.join(f"../logs", f"{current_time}")
    os.makedirs(log_name, exist_ok=True)
    os.makedirs(f"../logs/{current_time}/ckpts", exist_ok=True)
    print(f"Create new log: {log_name}")
    ## also copy the training configs from original config
    copy_file(src_file=args.train_config, dest_dir=f"{log_name}", new_filename=f"train-config.yaml")
    print(f"Saved train-config to {log_name}/train-config.yaml")
    
    # prepare dataset
    print(f"\nPreparation of Dataset Begin :(")
    trainset = MMDemos_small(
        path=train_config['data']['path'],
        handler=train_config['data']['handler'],
        train_split=train_config['data']['train_split'],
        multiplier=train_config['data']['multiplier'],
        seed=train_config['data']['seed'],
        seg_len=train_config['data']['seg_len'],
        init_mode=train_config['data']['init_mode'],
    )
    print(f"after trainset, trainset.n_modal: {trainset.n_modal}")
    print(f"after trainset, trainset.modal_dims: {trainset.modal_dims}")
    print(f"Preparation of Dataset End :)\n")
    
    # Create trainloader
    print(f"\nCreate trainloader :(")
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=train_config['train']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=train_config['train']['num_workers'],
        persistent_workers=True,
        drop_last=True,
    )
    print(f"Create trainloader End :)\n")
    
    # Config MInfoCon's Network
    ## configuer block size
    print(f"\nInitial Model :(")
    block_size = train_config['data']['seg_len']
    block_size = block_size if block_size is not None else trainset.max_traj_len
    block_scale = 1
    
    print(f"\nInitial Encoder :(")
    enc_config = MNConfig(n_embd=train_config['model']['n_embd'],
                          n_head=train_config['model']['n_head'],
                          attn_pdrop=train_config['model']['dropout'],
                          resid_pdrop=train_config['model']['dropout'],
                          embd_pdrop=train_config['model']['dropout'],
                          block_size=block_size,
                          block_scale=block_scale,
                          use_causal=train_config['model']['encoder']['use_causal'],
                          n_layer=train_config['model']['encoder']['n_layer'],
                          max_timestep=trainset.max_traj_len,
                          hyper_dim=train_config['model']['encoder']['hyper_dim'])
    print(f"Initial Encoder End :)\n")
    
    print(f"\nInitial Reconstructor :(")
    rec_config = MNConfig(n_embd=train_config['model']['n_embd'],
                          n_head=train_config['model']['n_head'],
                          attn_pdrop=train_config['model']['dropout'],
                          resid_pdrop=train_config['model']['dropout'],
                          embd_pdrop=train_config['model']['dropout'],
                          block_size=block_size,
                          block_scale=block_scale,
                          use_causal=train_config['model']['reconstructor']['use_causal'],
                          n_layer=train_config['model']['reconstructor']['n_layer'],
                          max_timestep=trainset.max_traj_len,
                          hyper_dim=train_config['model']['reconstructor']['hyper_dim'])
    print(f"Initial Reconstructor End :)\n")
    
    print(f"\nInitial Goal-Predictor :(")
    goal_config = MNConfig(n_embd=train_config['model']['n_embd'],
                           n_head=train_config['model']['n_head'],
                           attn_pdrop=train_config['model']['dropout'],
                           resid_pdrop=train_config['model']['dropout'],
                           embd_pdrop=train_config['model']['dropout'],
                           block_size=block_size,
                           block_scale=block_scale,
                           use_causal=train_config['model']['goal']['use_causal'],
                           n_layer=train_config['model']['goal']['n_layer'],
                           max_timestep=trainset.max_traj_len,
                           hyper_dim=train_config['model']['goal']['hyper_dim'])
    print(f"Initial Goal-Predictor End :)\n")
    
    # Configure Optimizer & Schedular
    ### Since there are some coes that weighting different loss
    ### and we also want them to change with times
    ### We will have a tricky way to load them from .yaml
    
    coes = {k: safe_handler(train_config['train']['coes'][k])
            for k in train_config['train']['coes'].keys()}
    ### test it a while
    plot_functions(coes)
    ### create CoefficientsAdjuster based on it
    optimizer_config = {
        'init_lr': train_config['train']['init_lr'],
        'weight_decay': train_config['train']['weight_decay'],
        'beta1': train_config['train']['beta1'],
        'beta2': train_config['train']['beta2'],
        'coes': CoefficientsAdjuster(coes)
    }
    ### test again
    plot_functions(optimizer_config['coes'].coefficient_functions)
    
    assert train_config['train']['lr_schedule'] in ['cos_decay_with_warmup', 'multistep', None], 'Unknown lr scheduler'
    if train_config['train']['lr_schedule'] == 'cos_decay_with_warmup':
        scheduler_config = {
            'type': 'cos_decay_with_warmup',
            't_max': train_config['train']['n_iters'],
            't_warmup': train_config['train']['t_warmup']
        }
    elif train_config['train']['lr_schedule'] == 'multistep':
        scheduler_config = {
            'type': 'multistep',
            'milestones': train_config['train']['milestones'],
            'gamma': train_config['train']['gamma']
        }
    else:
        scheduler_config = None
        
    
    # Create a model
    print(trainset.modal_dims)
    os.makedirs(f"../logs/{current_time}/metrics", exist_ok=True)
    hminfocon_model = HMInfoCon(enc_config=enc_config,
                                rec_config=rec_config,
                                goal_config=goal_config,
                                th_config=train_config['model']['th_config'],
                                opt_config=optimizer_config,
                                sch_config=scheduler_config,
                                modal_dims=trainset.modal_dims,
                                modal_usage=train_config['model']['modal_usage'],
                                modes=train_config['data']['loss_modes'],
                                name=f"{current_time}",
                                metric_path=f"../logs/{current_time}/metrics")


    # Setting ckpt saving strategy
    ### save every epoch
    ### also save the last checkpoint
    mysavelogger = MySaveLogger(path=f"../logs/{current_time}/ckpts",
                                iter_freq=train_config['train']['save_ckpt_every'])
    
    # Now we can start training
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[mysavelogger],
                         max_steps=train_config['train']['n_iters'])
    trainer.fit(model=hminfocon_model,
                train_dataloaders=trainloader)
