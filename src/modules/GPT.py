"""
Code for the module architecture of CoTPC, based on the GPT implementation.
Some of the key hyper-parameters are explained in GPTConfig.

References:
(1) https://github.com/karpathy/minGPT
(2) https://github.com/kzl/decision-transformer
"""

import sys
from itertools import accumulate
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .module_util import MLP, GELU


class SelfAttention(nn.Module):
    """
    Self Attn Layer, without causal
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        
    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalSelfAttention(nn.Module):
    """
    Self Attn Layer, can choose to be of scale * blocksize
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # scale block size based on config.block_size
        # register mask so that it can be more quicker when training
        block_size = config.block_size * config.block_scale
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        
        self.block_size = block_size
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        assert T <= self.block_size,f"Input sequence length ({T}) larger than block_size ({self.block_size})"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))  # Masked attention

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """
    A Transformer block,
    Can be Fully-connect or Forward-causal
    """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.use_causal:
            self.attn = CausalSelfAttention(config)
        else:
            self.attn = SelfAttention(config)
        
        if config.hyper_dim is not None:
            print(type(config.hyper_dim), config.hyper_dim)
            print(config.n_embd + config.hyper_dim)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd + config.hyper_dim, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )

    def forward(self, x, hyper_ctrl=None):
        x = x + self.attn(self.ln1(x))
        if hyper_ctrl is not None:
            # use a hyper control here, better conditioning
            x = x + self.mlp(torch.cat([self.ln2(x), hyper_ctrl], dim=-1))
        else:
            x = x + self.mlp(self.ln2(x))
        return x


class BlockLayers(nn.Module):
    """
    A wrapper class for a sequence of Transformer blocks
    Can be Fully-connect or Forward-causal
    """

    def __init__(self, config):
        super().__init__()
        # Register all the individual blocks.
        self.block_list = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.n_head = config.n_head

    def forward(self, x, hyper_ctrl=None):
        # B, T, _ = x.shape
        output = []  # Also keep the intermediate results.

        for block in self.block_list:
            x = block(x, hyper_ctrl)
                
        return x


class MNConfig:
    def __init__(self, n_embd, n_head,
                 attn_pdrop, resid_pdrop, embd_pdrop,
                 block_size, block_scale, use_causal, n_layer, max_timestep,
                 hyper_dim=None):
        self.n_embd = n_embd
        self.n_head = n_head
        
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        
        self.block_size = block_size
        self.block_scale = block_scale
        
        self.use_causal = use_causal
        
        self.n_layer = n_layer
        self.max_timestep = max_timestep
        
        self.hyper_dim = hyper_dim


class MNNet(nn.Module):
    """
    Input a sequence concatenating sequential data from M sensors
    Output a sequence concatenating sequential data from N sensors
    Also, we are able to select a subset of sensors, and mask all the inputs
    """
    def __init__(self, config, in_dim_list=None, out_dim_list=None, use_global=True, extra_emb=True):
        super().__init__()
        assert in_dim_list is not None and out_dim_list is not None
        assert all(isinstance(in_d, int) for in_d in in_dim_list)
        assert all(isinstance(out_d, int) for out_d in out_dim_list)
        
        self.config = config
        
        self.in_dim_list = in_dim_list
        self.n_sensor = len(in_dim_list)
        
        self.out_dim_list = out_dim_list
        self.n_output = len(out_dim_list)
        
        self.hid_dim = config.n_embd
        
        self.block_size = config.block_size * config.block_scale
        
        # mask for input, will be zero
        # input embeddings
        self.init_in_mlps(in_dim_list, config.n_embd)
        
        # time-step embeddings
        self.local_t_emb = nn.Parameter(torch.zeros(1, self.block_size, config.n_embd))
        # self.use_global = use_global
        # if use_global:
        #     self.global_t_emb = nn.Parameter(torch.zeros(1, config.max_timestep, config.n_embd))
        self.extra_emb = extra_emb
        
        # embedding dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer (attention layers)
        self.blocks = BlockLayers(config)
        self.ln = nn.LayerNorm(config.n_embd)
        
        # out prediction
        self.init_out_mlps(out_dim_list, config.n_embd)
        
        self.apply(self._init_weights)
        
    
    def init_in_mlps(self, in_dim_list, n_embd):
        self.e_ins = nn.ModuleList([MLP(in_dim, n_embd, [4*n_embd]) for in_dim in in_dim_list])
    
    
    def encode_input(self, in_list):
        in_xs = torch.stack([mlp(x) for mlp, x in zip(self.e_ins, in_list)], dim=0)  # [(B,T,n_embd), (B,T,n_embd), ...]
        in_x = torch.sum(in_xs, dim=0)
        return in_x
    
    
    def init_out_mlps(self, out_dim_list, n_embd):
        self.d_outs = nn.ModuleList([MLP(n_embd, out_dim, [4*n_embd, 4*n_embd]) for out_dim in out_dim_list])
    
    def decode_output(self, x):
        outs = [mlp(x) for mlp in self.d_outs]  # [(B,T,C1), (B,T,C2), ...]
        return outs
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, in_list, timesteps=None, mask_id=[], extra_emb=None):
        # mask input
        # assert len(mask_id) == len(in_list),f"len(mask_id): {len(mask_id)}, len(in_list): {len(in_list)}"
        for id in mask_id:
            in_list[id] = torch.zeros_like(in_list[id])
            
        # fuse multi-modality input
        token_embeddings = self.encode_input(in_list)   # BxTx{config.n_embd}
        B, T = tuple(token_embeddings.shape[:2])
        
        # time-step embeddings
        ### local time-step embeddings
        local_t_emb = self.local_t_emb[:, :T, :]
        t_emb = local_t_emb
            
        # extra embedding (for implicit hierarichal generative informativeness)
        if self.extra_emb:
            # print('using extra_emb')
            assert isinstance(extra_emb, torch.Tensor),"Please give torch.Tensor extra_emb"
            assert extra_emb.shape[-1] == token_embeddings.shape[-1],f"hope extra_emb.shape {token_embeddings.shape}, got {extra_emb.shape}"
            x = token_embeddings + t_emb + extra_emb
        else:
        # input ready for Transformer Blocks
            x = token_embeddings + t_emb
        
        # Transformer!!!
        x = self.drop(x)
        x = self.blocks(x, hyper_ctrl=extra_emb)
        x = self.ln(x)
        
        # output decoder
        out_list = self.decode_output(x)
        
        return out_list
