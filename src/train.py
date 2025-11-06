import os, sys, shutil
sys.path.append(os.path.abspath('../minfoncon/src'))
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import yaml
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

import pytorch_lightning as pl

from data import MMDemos_small
from modules.VQ import VQConfig, SinkhormKnoppConfig, KMeansConfig, EMAConfig, TrainTimeMemoConfig
from modules.GPT import MNConfig
from modules.coe_adjust import CoefficientsAdjuster

try:
    from minfocon import MInfoCon
    print("Import MInfoCon successful")
except ImportError:
    print("Import MInfoCon failed")

from lr_scheduler import CosineAnnealingLRWarmup
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import MySaveLogger
from util import safe_handler, plot_functions, copy_file


if __name__ == "__main__":
    # load .yaml config file from ../train-configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, help="path of config file")
    # parser.add_argument('--model_name', type=str, help="Model name from previous stage. Helpful for pipe in bash")
    # parser.add_argument('--ckpt', type=int, help="Corresponding checkpoint. Helpful for pipe in bash")
    args = parser.parse_args() # args only have a path
    print(args)
    
    # Load the hyperparameters from the path
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)
    print(train_config)
    
    # create a log file to save everything during training
    ## name is based on system time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_stage = train_config['model']['stage']
    log_name = os.path.join(f"../logs", f"stage-{train_stage}/stage-{train_stage}_{current_time}")
    os.makedirs(log_name, exist_ok=True)
    os.makedirs(f"../logs/stage-{train_stage}/stage-{train_stage}_{current_time}/ckpts", exist_ok=True)
    print(f"Create new log: ../logs/{log_name}")
    ## also copy the training configs from original config
    copy_file(src_file=args.train_config, dest_dir=f"../logs/{log_name}", new_filename=f"train-config.yaml")
    print(f"Saved train-config to ../logs/{log_name}/train-config.yaml")
    
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
        optic=train_config['data']['optic'],
    )
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
                          hyper_dim=train_config['model']['encoderr']['hyper_dim'])
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
    
    print(f"\nInitial Next-Predictor :(")
    next_config = MNConfig(n_embd=train_config['model']['n_embd'],
                           n_head=train_config['model']['n_head'],
                           attn_pdrop=train_config['model']['dropout'],
                           resid_pdrop=train_config['model']['dropout'],
                           embd_pdrop=train_config['model']['dropout'],
                           block_size=block_size,
                           block_scale=block_scale,
                           use_causal=train_config['model']['next']['use_causal'],
                           n_layer=train_config['model']['next']['n_layer'],
                           max_timestep=trainset.max_traj_len,
                           hyper_dim=train_config['model']['next']['hyper_dim'])
    print(f"Initial Next-Predictor End :)\n")
    
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
    
    print(f"\nInitial Codebook :(")
    ema_config = EMAConfig(coe=train_config['model']['codebook']['ema']['coe'])
    sk_config = SinkhormKnoppConfig(coe_h=train_config['model']['codebook']['sk']['coe_h'],
                                    n_iter=train_config['model']['codebook']['sk']['n_iter'])
    km_config = KMeansConfig(n_iter=train_config['model']['codebook']['km']['n_iter'],
                             pool_size=train_config['model']['codebook']['km']['pool_size'])
    ttm_config = TrainTimeMemoConfig(pool_size=train_config['model']['codebook']['ttm']['pool_size'])
    
    code_config = VQConfig(dim=train_config['model']['n_embd'],
                           N=train_config['model']['codebook']['N'],
                           usage=train_config['model']['codebook']['usage'],
                           ema_config=ema_config,
                           sk_config=sk_config,
                           km_config=km_config,
                           ttm_config=ttm_config,
                           preserve_grad=train_config['model']['codebook']['preserve_grad'])
    print(f"Initial Codebook End :)\n")
    
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
    os.makedirs(f"../logs/stage-{train_stage}/stage-{train_stage}_{current_time}/metrics", exist_ok=True)
    minfocon_model = MInfoCon(enc_config=enc_config,
                              rec_config=rec_config,
                              next_config=next_config,
                              goal_config=goal_config,
                              code_config=code_config,
                              opt_config=optimizer_config,
                              sch_config=scheduler_config,
                              modal_dims=trainset.modal_dims,
                              modes=train_config['data']['loss_modes'],
                              name=f"stage-{train_stage}_{current_time}",
                              metric_path=f"../logs/stage-{train_stage}/stage-{train_stage}_{current_time}/metrics")
    
    # load from checkpoints
    ### check from train_config and commandline input
    if train_config['model']['pretrain'] is not None and train_config['model']['ckpt'] is not None:
        pretrain_stage = train_config['model']['pretrain'][:7]
        pretrain_timestamp = train_config['model']['pretrain'][7:]
        ckpt_dir = f"../logs/stage-{pretrain_stage}/{pretrain_timestamp}/ckpts"
        ckpt_id = train_config['model']['ckpt']
        # load pretrained parameters
        minfocon_model.load_state_dict(torch.load(f"{ckpt_dir}/{ckpt_id}.pth"), strict=True)
        print(f"[stage-{train_stage}_{current_time}]: Load pretrained parameter from {ckpt_dir}/{ckpt_id}.pth")
        
    else:
        assert train_config['model']['pretrain'] is None and train_config['model']['ckpt'] is None
        print(f"[stage-{train_stage}_{current_time}]: Train from scratch.")
    
    # Setting ckpt saving strategy
    ### save every epoch
    ### also save the last checkpoint
    mysavelogger = MySaveLogger(path=f"logs/stage-{train_stage}/stage-{train_stage}_{current_time}/ckpts",
                                iter_freq=train_config['train']['save_ckpt_every'])
    
    
    # Now we can start training
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         callbacks=[mysavelogger],
                         max_steps=train_config['train']['n_iters'])
    trainer.fit(model=minfocon_model,
                train_dataloaders=trainloader)
