"""
Reference: https://github.com/ritheshkumar95/pytorch-vqvae/blob/master/functions.py
"""

import os
import sys
from typing import Optional
# sys.path.append(os.path.abspath('../modules'))
from math import log
import queue
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange

from .module_util import MLP, TimeSphereEncoder, FreqEncoder, print_vector, pca_visualize_prototypes, proto_specific_visualize


def FPSInit(vecs, K):
    # select K vectors from vecs [N, d] (all norm_2 == 1)
    # using FPS principle
    
    # random start
    min_k0 = torch.randint(low=0, high=K, size=(1,))
    # print(min_k0)
    select_vec = vecs[min_k0:min_k0+1]
    select_idx = torch.tensor([min_k0], device=vecs.device, dtype=torch.int32)
    
    for _ in range(1, K, 1):
        cos_sim = torch.mm(select_vec, vecs.permute(1, 0))  # [?, N]
        # print(cos_sim.shape)
        max_cos_sim = torch.max(cos_sim, dim=0).values      # [N]
        # print(max_cos_sim.shape)
        min_k = torch.min(max_cos_sim, dim=0).indices       # int
        select_idx = torch.cat([select_idx, torch.tensor([min_k], device=vecs.device, dtype=torch.int32)], dim=0)
        # print(min_k)
        select_vec = torch.cat([select_vec, vecs[min_k:min_k+1]], dim=0)
    
    return select_vec, select_idx


@torch.no_grad()
def SphericalKMeans(vecs, K, n_iters=20, use_km=True):
    if torch.isnan(vecs).any():
        print("Assert Error in [SphericalKMeans]:")
        print(f'!!!!! vecs (after FPS) have a tensor with NaN value !!!!!')
        assert False
    
    kmeans_init, select_idx = FPSInit(vecs, K)   # [K, d], initialization of prototypes    
    if torch.isnan(kmeans_init).any():
        print("Assert Error in [SphericalKMeans]:")
        print(f'!!!!! kmeans (after FPS) have a tensor with NaN value !!!!!')
        assert False
    kmeans = kmeans_init
    
    if not use_km:
        return kmeans
    
    for iter in range(n_iters):
        # Updata Selection of each prototype
        kmeans_old = kmeans
        cos_sim = torch.mm(kmeans, vecs.permute(1, 0))  # [K, N]
        # print(cos_sim.shape)
        max_cos_sim = torch.max(cos_sim, dim=0, keepdim=True).indices  # [1, N]
        max_map = (max_cos_sim == torch.arange(K, device=vecs.device).unsqueeze(1)).to(dtype=torch.float32) # [K, N]
        # to avoid overrange
        # print(torch.sum(max_map, dim=1, keepdim=True).shape)
        # input()
        max_map_cnt = torch.sum(max_map, dim=1, keepdim=True)
        max_map_cnt_flat = max_map_cnt.view(-1)
        
        max_map = torch.div(max_map, 1 + max_map_cnt)
        if torch.isnan(max_map).any():
            print("Assert Error in [SphericalKMeans]:")
            print(f'!!!!! max_map (at iter {iter} after div) have a tensor with NaN value !!!!!')
            assert False
        # print(max_map.shape)
        # Updata prototypes
        kmeans = F.normalize(torch.mm(max_map, vecs), p=2.0, dim=-1)
        # check norm
        kmeans_norm_flat = torch.norm(kmeans, p=2.0, dim=-1)
        
        if torch.min(kmeans_norm_flat) <= 0.2:
            print("too small kmeans_norm at iter {iter}, something happened")
            print(f'min norm_2 of kmeans: {torch.min(kmeans_norm_flat)}')
            print(f'max norm_2 of kmeans: {torch.max(kmeans_norm_flat)}')
            # check initialization
            print('### Initialization  SELECTION ###')
            for tmpi, idx in enumerate(select_idx):
                print(f'Initialize\tConcept_{tmpi}\tusing vec_{idx}')
            kmeans_init_norm = torch.norm(kmeans_init, p=2.0, dim=-1)
            for tmpi, n in enumerate(kmeans_init_norm):
                print(f'Initialized \tConcept_{tmpi}\tnorm_2:\t{n}')
            # check cnt
            print('### Concept CNT ###')
            for tmpi, cnt in enumerate(max_map_cnt_flat):
                print(f'|Concept_{tmpi}|\t=\t{cnt}')
            print('########## Concept CNT ##########')
            
            assert False
        
        if torch.isnan(kmeans).any():
            print("Assert Error in [SphericalKMeans]:")
            print(f'!!!!! kmeans (at iter {iter}) have a tensor with NaN value !!!!!')
            assert False
        # print(kmeans.shape)
        # print(F.cosine_similarity(kmeans_old, kmeans))
        
        
    return kmeans
    


def SinkHornAlgorithm(scores, eps=0.05, n_iters=3):
    # print(f"scores.shape: {scores.shape}")
    
    Q = torch.exp(torch.div(scores, eps)).permute(1, 0)
    # print(f"Q.shape: {scores.shape}")
    
    Q = torch.div(Q, torch.sum(Q))
    Q = torch.nan_to_num(Q, nan=0.0, posinf=1e8, neginf=-1e8)
    
    K, B = Q.shape
    
    u = torch.zeros((K,), device=scores.device)
    r = torch.full((K,), fill_value=1/K, device=scores.device)
    c = torch.full((B,), fill_value=1/B, device=scores.device)
    
    for _ in range(n_iters):
        u = torch.sum(Q, dim=1)
        
        Q = torch.mul(Q, (torch.div(r, u)).unsqueeze(1))
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e8, neginf=-1e8)
        
        Q = torch.mul(Q, (torch.div(c, torch.sum(Q, dim=0))).unsqueeze(0))
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e8, neginf=-1e8)
    
    Q = (torch.div(Q, torch.sum(Q, dim=0, keepdim=True))).permute(1, 0)
    Q = torch.nan_to_num(Q, nan=0.0, posinf=1e8, neginf=-1e8)
        
    return Q  # (B, K)
    

def awe_contiguous_indices(indices, distance):
    # indices [B, T] int
    # distance [B, T] float
    B, T = indices.shape
    indices_new = [indices[:, 0:1]]
    distance_t = distance[:, 0:1]
    for t in range(1, T, 1):
        distance_t = distance_t - 1
        # keep same indices whtn distance_t > 0
        indice_new = torch.where(distance_t >= 0, indices_new[-1], indices[:, t:(t+1)])
        indices_new.append(indice_new)
        distance_t = torch.where(distance_t >= 0, distance_t, distance[:, t:(t+1)])
    indices_new = torch.cat(indices_new, dim=1)
    # print('indices_new\n', indices_new)
    
    return indices_new


# class increaseEncourageIdentityMap(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input

#     @staticmethod
#     def backward(ctx, grad_output):
#         mask_increase = torch.less_equal(grad_output, 0.0)
#         grad_scale = torch.where(mask_increase, 10.0, 1.0)
#         return grad_output * grad_scale

# increae_encourage_identity_map = increaseEncourageIdentityMap.apply


class EMAConfig:
    def __init__(self, coe=0.99):
        self.coe = coe


class SinkhormKnoppConfig:
    def __init__(self, coe_h=0.05, n_iter=3):
        self.coe_h = coe_h
        self.n_iter = n_iter


class KMeansConfig:
    def __init__(self, n_iter=20, pool_size=10_000_000):
        self.n_iter = n_iter
        self.pool_size = pool_size


class TrainTimeMemoConfig:
    def __init__(self, pool_size):
        self.pool_size = pool_size
     

class VQConfig:
    def __init__(self, dim, N, usage, ema_config, sk_config, km_config, ttm_config, preserve_grad):
        self.dim = dim
        self.N = N
        self.usage = usage
        self.ema_config = ema_config
        self.sk_config = sk_config
        self.km_config = km_config
        self.ttm_config = ttm_config
        self.preserve_grad = preserve_grad


class VQ_Sinkhorn_Knopp(nn.Module):
    def __init__(
        self,
        dim, N,
        usage=0.9999,
        ema_config=None,
        sk_config=None,
        km_config=None,
        ttm_config=None,
        preserve_grad='ave',
        mode='train',
    ):
        super().__init__()
        
        self.dim = dim  # dimension of each code item
        self.N = N      # number of code in the codebook
        self.KT = 2.0 / (log((N - 1.0) * usage / (1.0 - usage)))    # temperature, since cos-sim is between [-1, 1], set proper value to make usage of softmax space 

        self.sm = nn.Softmax(dim=-1)

        arange = torch.arange(N, dtype=torch.float32)
        self.register_buffer('arange', arange)
        
        self.keys = nn.Embedding(N, dim)
        # Hyper-sphere initialization
        init.normal_(self.keys.weight, mean=0.0, std=1.0)
        with torch.no_grad():
            self.keys.weight.div_(self.keys.weight.norm(p=2, dim=1, keepdim=True))
        self.keys.requires_grad_(False)

        # parameters
        if mode == 'train':
            assert isinstance(ema_config, EMAConfig),f"Please properly set ema_config, your ema_config type: {type(ema_config)}"
            self.ema_config = ema_config
            assert isinstance(sk_config, SinkhormKnoppConfig),f"Please properly set sk_config, your sk_config type: {type(sk_config)}"
            self.sk_config = sk_config
            assert isinstance(km_config, KMeansConfig),f"Please properly set km_config, your km_config type: {type(km_config)}"
            self.km_config = km_config

            # how to preserve gradient
            assert preserve_grad in ['ave', 'direct'],"Unknown way to preserve gradient"
            self.preserve_grad_ave = (preserve_grad == 'ave')
            self.vec_bank = []
            self.vec_bank_len = 0

            # To handle over-smoothing
            # We need a memo to record activation of codebooks
            assert isinstance(ttm_config, TrainTimeMemoConfig),f"Please properly set ttm_config, your ttm_config type: {type(ttm_config)}"
            self.ttm_config = ttm_config
            ## create a FIFO based structure to record selection
            self.train_time_memo = []
        elif mode == 'eval':
            self.ema_config = ema_config
            self.sk_config = sk_config
            self.km_config = km_config
            self.preserve_grad_ave = preserve_grad
            self.ttm_config = ttm_config
            self.train_time_memo = None
            assert self.ema_config is None
            assert self.sk_config is None
            assert self.km_config is None
            assert self.preserve_grad_ave is None
            assert self.ttm_config is None
        else:
            print(f'Unknown train/eval mode: {mode}')
            assert False
            
    
    
    def insert_train_time_memo(self, statistic):
        # statistic (self.N), record the frequency of each code in codebook
        # return
        # print("insert")
        self.train_time_memo.append(statistic)
        # print(self.train_time_memo)
        if len(self.train_time_memo) > self.ttm_config.pool_size:
            # pop out first item
            self.train_time_memo.pop(0)
    
    
    def insert_vecs(self, vecs):
        # embedding vectors (B, d)
        self.vec_bank.append(vecs)
        self.vec_bank_len += 1  # actually batchsize...
        
        if self.vec_bank_len > self.km_config.pool_size:
            # pop out first item
            item = self.vec_bank.pop(0)
            self.vec_bank_len -= 1
    
    
    def reset_dead(self):
        # return
        # Reset unused items based on self.train_time_memo
        ## Get zero and unzero statistic
        # print(torch.stack(self.train_time_memo, dim=0))
        statistic_sum = torch.sum(torch.stack(self.train_time_memo, dim=0), dim=0)
        sum_all = torch.sum(statistic_sum)
        # print(f"statistic_sum: {statistic_sum}")
        # print(f"sum_all: {sum_all}")
        none_zero_idx = (statistic_sum > 0.1 * sum_all // (self.N)).nonzero(as_tuple=True)[0]
        # print(none_zero_idx)
        # input()
        if none_zero_idx.shape[0] == self.N:
            return
        else:
            zero_idx = (statistic_sum <= 0.1 * sum_all // (self.N)).nonzero(as_tuple=True)[0]
            # print(torch.argsort(statistic_sum[none_zero_idx], descending=True))
            # print(f"none_zero_idx: {none_zero_idx}")
            # print(f"zero_idx: {zero_idx}")
            none_zero_idx = none_zero_idx[torch.argsort(statistic_sum[none_zero_idx], descending=True)]
            # print(f"none_zero_idx: {none_zero_idx}")
            # cycle extend
            print("Reset:", zero_idx)
            print("From:", none_zero_idx)
            
            none_zero_idx = none_zero_idx.repeat((zero_idx.shape[0] + none_zero_idx.shape[0] - 1) // none_zero_idx.shape[0])[:zero_idx.shape[0]]
            # print(f"none_zero_idx: {none_zero_idx}")
            # charge into the crowds
            self.keys.weight.data[zero_idx] = self.keys.weight.data[none_zero_idx]
            return
    
    def reset_kmeans(self):
        vecs = torch.cat(self.vec_bank, dim=0)
        print(f'before reset_kmenas: vecs.shape={vecs.shape}')
        self.keys.weight.data = SphericalKMeans(
            vecs=torch.cat(self.vec_bank, dim=0),
            K = self.N,
            n_iters=self.km_config.n_iter
        )
    
    def dump_norm(self, data):
        # print("Norm of codebook")
        # print_vector(self.keys.weight.norm(p=2, dim=1), "float")
        pca_visualize_prototypes(data, self.keys.weight.data)
        
    def dump_dis(self, data, assign):
        proto_specific_visualize(data, self.keys.weight.data, assign)
    
    
    def normalize_keys(self):
        # DO I NEED TO STOP GRADIENT HERE??? YES!!! BECAUSE I USE EMA TO MOVE PROTOTYPES
        with torch.no_grad():
            self.keys.weight.div_(self.keys.weight.norm(p=2, dim=1, keepdim=True))
    
    
    def get_score(self, input, keys):
        # input shape (B*T, dim)
        
        # Calculate score
        input = F.normalize(input, p=2.0, dim=-1)   # (B*T, dim)
        score = torch.mm(input, keys.permute(1, 0).detach()) # (B*T, dim) @ (dim, N) -> (B*T, N)
        
        # temperature
        score = torch.div(score, self.KT)
        
        return score            
    
    
    def softmax_assign(self, score):
        # score shape (B*T, N)
        P = self.sm(score)
        P = torch.nan_to_num(P)
        return P
        
    
    def sinkhorn_assign(self, score):
        # input score
        Q = SinkHornAlgorithm(score, eps = self.sk_config.coe_h, n_iters=self.sk_config.n_iter) # (B*T, N)
        return Q
    

    def get_indices(self, input):
        # input shape (B, T, key_dim)
        
        with torch.no_grad():   # choosing indices does not need gradient
            input = input.contiguous()
            B, T = input.shape[0], input.shape[1]
            input_f = input.view(B*T, self.dim)  # (B*T, dim)
            
            # normalize keys
            self.normalize_keys()
            
            # get score
            score_f = self.get_score(input=input_f, keys=self.keys.weight)    # (B*T, N)
            P_f = self.softmax_assign(score_f) # (B*T, N)
            P = P_f.view(B, T, -1)

            # use the nearest key
            indices_f = torch.argmax(score_f, dim=1)  # (B*T)
            indices = indices_f.view(B, T)    # (B, T)
            
            # get embedding
            indices_emb = self.keys.weight[indices] # (B, T, dim)

            return indices, indices_emb, P


    def get_w_cnt(self, indices):
        # indices shape (B, T)
        
        indices_f = indices.view(-1)    # (B*T)
        
        # count number of indice for each indice in [0, 1, ..., (N-1)]
        sum_mask = torch.eq(indices_f.unsqueeze(0), self.arange.unsqueeze(1))   # (N, B*T)
        cnt = torch.add(torch.sum(sum_mask, dim=1), 1.0)    # (N)
        w = torch.div(1.0, cnt) # (N), because we need to get average afterwards
        w = torch.nan_to_num(w)
        
        # get average coe.
        w_cnt = w[indices]      # (B, T), average coe. at every (B, T) position
        return w_cnt


    def training_forward(self, input: torch.Tensor, lr):
        # input shape (B, T, dim)
        
        input = input.contiguous()
        
        if torch.isnan(input).any():
            print(f'!!!!! input have a tensor with NaN value !!!!!')
            assert False
        
        B, T = input.shape[0], input.shape[1]
        input_f = input.view(B*T, self.dim) # (B*T, dim)
        
        # check norm of self.keys.weight.data
        # key_norm_flat = self.keys.weight.norm(p=2, dim=1, keepdim=True).view(-1)
        # for tmpi, n in enumerate(key_norm_flat):
        #     print(f'norm_2 of concept #{tmpi}: {n}')
        
        if torch.isnan(self.keys.weight.data).any():
            print(f'!!!!! keys (before norm) have a tensor with NaN value !!!!!')
            assert False
        self.normalize_keys()       # normalize keys
        keys = self.keys.weight.data.detach()    # stop gradient of keys
        if torch.isnan(keys).any():
            print(f'!!!!! keys (after norm) have a tensor with NaN value !!!!!')
            assert False
        
        # Also, above keys is the copy of keys
        # We need a copy of this, because later we will update the original keys using EMA
        
        # get score
        score_f = self.get_score(input=input_f, keys=keys) # (B*T, N)
        
        # different distribution
        P = self.softmax_assign(score_f)  # (B*T, N), Based on softmax
        if torch.isnan(P).any():
            print(f'!!!!! P have a tensor with NaN value !!!!!')
            assert False
        
        Q = self.sinkhorn_assign(score_f).detach() # (B*T, N), Based on Sinkhorn-Knopp
        if torch.isnan(Q).any():
            print(f'!!!!! Q have a tensor with NaN value !!!!!')
            assert False
        
        # Q = P.detach()
        
        # indices assignment
        ### indices output at time, based on P actually
        indices_f = torch.argmax(P, dim=1)    # (B*T)
        indices = indices_f.view(B, T)        # (B, T)
        ### indices output heading, based on Q
        indices_next_f = torch.argmax(Q, dim=1)     # (B*T)
        # indices_next = indices_next_f.view(B, T)    # (B, T)
        
        # First prepare output for further calculation
        indices_emb_f = keys[indices_f] # (B*T, dim)
        ### IMPORTANT, preserve gradient
        if self.preserve_grad_ave:  # use average of keys to preserve gradient, since P have gradient of input inside
            indices_emb_s = torch.mm(P, keys)   # (B*T, dim)
            if torch.isnan(indices_emb_s).any():
                print(f'!!!!! indices_emb_s have a tensor with NaN value !!!!!')
                assert False
            
            indices_emb_f = indices_emb_s + (indices_emb_f - indices_emb_s).detach()    # (B*T, dim)
        else:   # directly copy gradient back to input
            indices_emb_f = input_f + (indices_emb_f - input_f).detach()    # (B*T, dim)
        indices_emb = indices_emb_f.view(B, T, self.dim)    # (B, T, dim)
        
        # Then we prepare loss and related calculation to update codebook & assignment
        loss_emb2proto = F.cross_entropy(P, Q) # use this part to update softmax selection of input
        if torch.isnan(loss_emb2proto).any():
            print(f'!!!!! loss_emb2proto have a tensor with NaN value !!!!!')
            assert False
        
        ### Finally we use EMA to update keys behind, based on Q,
        ### and now we use the [self.keys] instead of the copy [keys]
        with torch.no_grad():
            # count number of indice for each indice in [0, 1, ..., (N-1)]
            sum_mask = torch.eq(self.arange.unsqueeze(1), indices_next_f.unsqueeze(0)).to(dtype=torch.float32)
            cnt = torch.add(torch.sum(sum_mask, dim=1, keepdim=True), 1.0)  # (N, 1)
            # print("try insert")
            self.insert_train_time_memo(cnt.to(dtype=torch.int32).view(-1))
            self.insert_vecs(input_f.detach())
            ave_input = torch.div(torch.mm(sum_mask, input_f), cnt)  # (N, dim)
            ave_input = torch.nan_to_num(ave_input)  # (N, dim)
            ema_coe = self.ema_config.coe
            self.keys.weight.data = F.normalize(
                self.keys.weight.data * ema_coe + ave_input * (1.0 - ema_coe),
                p=2.0, dim=-1
            )   # (N, dim)
            if torch.isnan(self.keys.weight.data).any():
                print(f'!!!!! self.keys.weight.data have a tensor with NaN value !!!!!')
                assert False
            # else:
            #     print(f'!!!!! self.keys.weight.data do not have a tensor with NaN !!!!!')
            
        # now we can return our calculation results for further calculation
        return indices, indices_emb, loss_emb2proto
    
    
    def forward(self, input, lr):
        # input shape (B, T, dim)
        return self.training_forward(input, lr)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    plain_test = F.normalize(torch.randn(size=(10_000_000, 128), device='cuda'), p=2.0, dim=-1)
    print(plain_test)
    
    select_vec = SphericalKMeans(plain_test, 30)
    print(select_vec)
