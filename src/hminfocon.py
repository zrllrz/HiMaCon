import os, sys
sys.path.append(os.path.abspath('../src'))
import numpy as np
from math import exp, pow, log, cos, sin, pi, isnan, floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from modules.module_util import print_vector, print_dict, generate_non_zero_list, save_frequency_plot
from modules.VQ import VQ_Sinkhorn_Knopp, VQConfig
from modules.GPT import MNConfig, MNNet, MLP
from modules.coe_adjust import CoefficientsAdjuster

import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from lr_scheduler import CosineAnnealingLRWarmup

from util import get_norm_loss


class HMInfoCon(pl.LightningModule):
    def __init__(self,
                 enc_config: MNConfig,
                 rec_config: MNConfig,
                 goal_config: MNConfig,
                 th_config=1000,
                 optic=False,
                 opt_config=None,
                 sch_config=None,
                 modal_dims=None,
                 modal_usage=None,
                 modes=None,
                 name='default',
                 metric_path=None):
        
        super().__init__()
        
        self.model_name = name
        assert isinstance(metric_path, str),f'Please input a str log_path, now log_path type: {type(metric_path)}'
        self.metric_path = metric_path
        
        self.opt_config = opt_config
        self.sch_config = sch_config

        assert modal_dims is not None and modes is not None
        self.n_modal = len(modal_dims)
        assert modal_usage != 'None'
        # should be a mask with size of (self.n_modal)
        # 1 indicating using this modality
        # 0 indicating do not use this modality
        self.modal_usage = modal_usage
        
        print(f"self.modal_usage: {self.modal_usage}")
        
        self.modes = modes

        self.encoder = MNNet(config=enc_config,
                             in_dim_list=modal_dims,
                             out_dim_list=[enc_config.n_embd],
                             use_global=False,
                             extra_emb=False) # if enc_config is not None else None
        
        self.pred_now = MNNet(config=rec_config,
                              in_dim_list=[enc_config.n_embd] + modal_dims,
                              out_dim_list=modal_dims,
                              use_global=False,
                              extra_emb=False)
        
        self.pred_goal = MNNet(config=goal_config,
                               in_dim_list=[enc_config.n_embd] + modal_dims,
                               out_dim_list=modal_dims,
                               use_global=False,
                               extra_emb=False)
        
        # threshold, changing, using for selecting 'goal state'
        assert isinstance(th_config, int)
        self.N_th = th_config
        self.KT = 9.2
        self.iter_th = 0
        
        # possible th in
        # (1/N) * [0,1,2,...,N] \in \R\intersection[0, 1]
        self.iter_map = [((1+iii) / th_config) for iii in range(th_config)]
        assert len(self.iter_map) == th_config
        
        
        
        # whether use optic flow or not
        assert (optic == False)
        self.optic = optic
        
        # embeddings
        self.th_embedding = nn.Parameter(torch.zeros(self.N_th, enc_config.n_embd))
        
        if opt_config is not None:
            self.opt_config = opt_config
            self.coes = opt_config['coes']
            assert isinstance(self.coes, CoefficientsAdjuster)
            assert {'enc', 'now', 'goal', 'th'}.issubset(set(self.coes.coefficient_functions.keys()))
            self.coes_t = {k : 0.0 for k in self.coes.coefficient_functions.keys()}

            self.progress_bar_step = 0.0

            self.automatic_optimization = False
        else:
            assert sch_config is None
            print('only label')
        

    def on_train_batch_start(self, batch, batch_idx):
        max_steps = self.trainer.max_steps
        global_step = self.trainer.global_step
        self.progress_bar_step = global_step / max_steps
        self.coes_t = self.coes.adjust_coefficients(self.progress_bar_step)

    def dump_statistic(self, indices, z_soft):
        print(f"model name: {self.model_name}")
        print_dict(self.coes_t)
    
    # @torch.no_grad()
    def get_goal_state_tolerance(self, in_list, dis: torch.Tensor, th_distance):
        # in_list: [(B, T, D1), ..., (B, T, Dm)] of m modalities
        # dis: (B, T, T) map logging distance
        assert 0. <= th_distance <= 1.
        
        B, T = dis.shape[0], dis.shape[1]
        # feature similarity
        assert list(dis.shape) == [B, T, T]
    
        # then select based on th_epsilon
        cluster_map = (dis <= th_distance).to(dtype=torch.int32)
        
        # for latter, we will let the lower triangular matrix to be zero
        triu_map = torch.triu(torch.ones(T, T, device=dis.device, dtype=torch.int32)).expand(B, -1, -1)
        cluster_map = cluster_map * triu_map
        
        # then, we sum up in the following way
        # a_11 a_12 a_13    a_11 a_12+a_22 a_13+a_23+a_13
        # 0    a_22 a_23 -> 0    a_22      a_23+a_33
        # 0    0    a_33    0    0         a_33
        # Can be done by the following matrix transformation
        # --     --   --              --     --                             --
        # | 1 1 1 |   | a_11 a_12 a_13 |     | a_11 a_12+a_22 a_13+a_23+a_33 |
        # |   1 1 | @ |      a_22 a_23 | --> |      a_22      a_23+a_33      | 
        # |     1 |   |           a_33 |     |                a_33           |
        # --     --   --              --     --                             --
        sum_cluster_map = torch.bmm(triu_map.float(), cluster_map.float()).to(dtype=torch.int32)
        sum_total_map = torch.bmm(triu_map.float(), triu_map.float())  # .to(dtype=torch.int32)
        
        cluster_ratio = torch.div(sum_cluster_map.to(dtype=torch.float32), sum_total_map.to(dtype=torch.float32))
        cluster_ratio = torch.clip(cluster_ratio, min=0.0, max=1.0)
        
        new_sum_cluster_map = torch.zeros((B, T+1, T+1), dtype=torch.int32, device=dis.device)
        new_sum_cluster_map[:, :T, :T] = (cluster_ratio >= 1.0) # (B, T, T)
        
        # now we can get key state now
        t_begin = torch.zeros((B,), dtype=torch.int32, device=dis.device)
        arange_B = torch.arange(B, dtype=torch.int32, device=dis.device)
        mask = torch.zeros((B, T), dtype=torch.bool, device=dis.device)
        for t in range(1, T+1, 1):
            mask[:, t-1] = ~(new_sum_cluster_map[arange_B, t_begin, t]).bool()
            t_begin = torch.where(mask[:, t-1], torch.full_like(t_begin, fill_value=t), t_begin)
        
        if self.training:
            self.log_dict(
                {
                    'n_seg': torch.mean(torch.sum(mask, dim=1).float()),
                    'n_tol': th_distance
                }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
            )
        
        # locate goal states
        arange_ext = torch.arange(T, device=dis.device).view(1, -1).repeat(B, 1)
    
        # create last mask, do not apply loss on last part, since we may use sub-trajectories
        last_mask = mask[:, :-1] + torch.sum(mask[:, :-1], dim=-1, keepdim=True) - torch.cumsum(mask[:, :-1], dim=-1)
        last_mask = last_mask.to(dtype=torch.bool)
        last_mask = torch.cat([last_mask, torch.zeros(size=(B, 1), dtype=torch.bool, device=dis.device)], dim=-1)
        last_mask = last_mask.to(dtype=torch.float32)
        his_ind = torch.full(size=(B, 1), fill_value=(T-1), device=dis.device)
        future_state_indices = torch.full(size=(B, 1), fill_value=(T-1), device=dis.device)
        for i in range(T-2, -1, -1):
            new_ind = torch.where(mask[:, i:(i+1)], arange_ext[:, i:(i+1)], his_ind)
            his_ind = new_ind
            future_state_indices = torch.cat([new_ind, future_state_indices], dim=1)
        
        future_state_indices = torch.clip(future_state_indices+1, min=0, max=T-1)
        
        future_state = [
            torch.gather(
                x, dim=1,
                index=future_state_indices.unsqueeze(-1).repeat(1, 1, x.shape[2])
            ) for x in in_list
        ]
        return future_state, future_state_indices, last_mask
    
    
    def always_contrast(self, z_soft):
        # z_soft [B, T, d]
        B, T = z_soft.shape[0], z_soft.shape[1]
        idx_orgi = torch.arange(T, device=z_soft.device).unsqueeze(0).repeat(B, 1)
        assert idx_orgi.shape[0] == B and idx_orgi.shape[1] == T
        idx_rand = torch.fmod(idx_orgi + torch.randint(low=1, high=T, size=(B, T), device=z_soft.device), torch.tensor([[T]], device=z_soft.device))
        assert torch.max(idx_rand) <= T-1 and torch.min(idx_rand) >=0
        assert not torch.any(idx_orgi==idx_rand)
        
        # calculate contrastive loss
        z_soft_contrastive = z_soft[torch.arange(B, device=z_soft.device).unsqueeze(1), idx_rand]
        assert z_soft_contrastive.shape[0] == B and z_soft_contrastive.shape[1] == T
        
        sigmoid_cos_sim = F.sigmoid(torch.mul(F.cosine_similarity(z_soft, z_soft_contrastive, dim=-1), self.KT)) # (B, T)
        
        ls_contrastive = torch.log(1.0 - sigmoid_cos_sim)
        assert ls_contrastive.shape[0] == B and ls_contrastive.shape[1] == T
        
        return -ls_contrastive
        
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        # get input from batch
        timesteps, lengths = batch['t'], batch['lengths']
        in_list = batch['in'] # let dataset be a list of modalities in order
        
        # modality ablation mask
        assert self.modal_usage is not None
        for i in range(self.n_modal):
            if self.modal_usage[i] == 0:
                in_list[i] = torch.zeros_like(in_list[i])
        
        # smooth encoding
        z_soft_list = self.encoder(in_list, timesteps, [])
        z_soft = z_soft_list[0] # see init of self.encoder, always output n_embd size vectors
        if torch.isnan(z_soft).any():
            print(f'!!!!! z_soft (before normalization) have NaN value !!!!!')
            assert False
        B, T = z_soft.shape[0], z_soft.shape[1]
        
        z_soft = F.normalize(z_soft, p=2.0, dim=-1) # (B, T, d), on hyper-sphere now
        if torch.isnan(z_soft).any():
            print(f'!!!!! z_soft (after normalization) have NaN value !!!!!')
            assert False
        
        # contrastive every time-step
        ls_contrastive = self.always_contrast(z_soft)
        
        if torch.isnan(ls_contrastive).any():
            print(f'!!!!! ls_contrastive have NaN value !!!!!')
            assert False
        assert list(ls_contrastive.shape) == [B, T]
        
        l_contrastive = torch.mean(ls_contrastive)
        
        # now we are able to do the prediction
        if self.coes_t['now'] > 0.0:
            # randomly mask input to do prediction
            mask_id = generate_non_zero_list(self.encoder.n_sensor)
            mask_id = [mask_id_item + 1 for mask_id_item in mask_id]
            
            # do mask reconstruction
            list_pred_now = self.pred_now([z_soft] + in_list, timesteps, mask_id)
            
            l_nows = [
                get_norm_loss(pred_now, now, lengths, mode)
                for pred_now, now, mode in zip(list_pred_now, in_list, self.modes)
            ]
            l_now = sum(l_nows)
            
        else:
            l_now = 0.0
            
            
        # Goal prediction, Try to build time relationship
        if self.coes_t['goal'] > 0.0:
            # hard encoding, no need anymore
            # instead, we will use self.coes_t['th'] to decide prediction of "how far" future
            
            # first calculate similarity
            with torch.no_grad():
                sim = F.cosine_similarity(z_soft.unsqueeze(1), z_soft.unsqueeze(2), dim=-1)
                sim = torch.clip(sim, min=-1.0, max=1.0)
                arccos_dis = torch.div(torch.arccos(sim), 3.15)
        
                arccos_dis_min, arccos_dis_max = torch.min(arccos_dis), torch.max(arccos_dis)
                
                th_rand = torch.randint(low=floor(arccos_dis_min*self.N_th),
                                        high=max(floor(arccos_dis_max*self.N_th), floor(arccos_dis_min*self.N_th)+1),
                                        size=(1,),
                                        device=z_soft.device) # random sample from \in {0,1,...,(self.N_th-1)*self.coes_t['th']}
                th_rand = int(max(min(th_rand, self.N_th-1), 0))
                th_epsilon = self.iter_map[th_rand] # in \{  0, 1/N, 2/N, ..., floor[(self.N_th-1)*self.coes_t['th']]/N, \}
                th_epsilon = max(min(th_epsilon, 1.0), 0.0)
            
                assert arccos_dis_min <= th_epsilon <= arccos_dis_max
    
                # select goal state based on th_epsilon
                goal_list, _, _ = self.get_goal_state_tolerance(in_list, arccos_dis, th_epsilon)
        
            # select embedding based on th
            extra_emb = self.th_embedding[th_rand].view(1,1,-1).repeat(z_soft.shape[0], z_soft.shape[1], 1)
        
            # no mask, just predict it!
            list_pred_goal = self.pred_goal([z_soft] + in_list, timesteps, [], extra_emb)
        
            # loss of goal prediction
            l_goals = [
                get_norm_loss(pred_goal, goal, lengths, mode)
                for pred_goal, goal, mode in zip(list_pred_goal, goal_list, self.modes)
            ]
            if self.modal_usage is not None:
                l_goal = sum([l * w for l, w in zip(l_goals, self.modal_usage)])
            else:
                l_goal = sum(l_goals)
            l_goal = sum(l_goals)
            
        else:
            l_goal = 0.0
        
        if isnan(l_now):
            print(f'!!!!! l_now have NaN value !!!!!')
            assert False
        if isnan(l_goal):
            print(f'!!!!! l_goal have NaN value !!!!!')
            assert False
        if isnan(l_contrastive):
            print(f'!!!!! l_contrastive have NaN value !!!!!')
            assert False
        
        
        loss = \
            float(self.coes_t['enc']) * l_contrastive \
            + float(self.coes_t['now']) * l_now \
            + float(self.coes_t['goal']) * l_goal
            
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch.step()

        self.log_dict(
            {
                'no': l_now,
                'go': l_goal,
                'co': l_contrastive,
                'ls': loss,
                'eps': float(th_epsilon),
            }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )

    def label_single(self, in_list, timesteps):
        # states: (T, s_dim)
        # actions: None or (T - 1, a_dim)
        # timesteps
        
        # smooth encoding
        z_soft_list = self.encoder(in_list, timesteps, [])
        z_soft = z_soft_list[0] # see init of self.encoder, always output n_embd size vectors
        z_soft = F.normalize(z_soft, p=2.0, dim=-1)
        
        return z_soft

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the module into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()  # all parameters which need to decay
        no_decay = set()  # all parameters which do not need to decay

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    print(fpn)
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)


        # special case the position embedding parameter in the root GPT module as not decayed
        # NEED MODIFICATION
        no_decay.add('encoder.local_t_emb')
        no_decay.add('pred_now.local_t_emb')
        no_decay.add('pred_goal.local_t_emb')
        no_decay.add('pred_goal.local_t_emb')
        no_decay.add('th_embedding')
        

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params),)

        optim = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))],
             "weight_decay": self.opt_config['weight_decay']},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        if self.opt_config is not None:
            optimizer = torch.optim.AdamW(
                optim,
                lr=self.opt_config['init_lr'],
                betas=(self.opt_config['beta1'], self.opt_config['beta2'])
            )
        else:
            optimizer = torch.optim.AdamW(
                optim,
                lr=1e-4,
                betas=(0.9, 0.95)
            )
        
        # scheduler config
        if self.sch_config is None:
            return optimizer

        assert 'type' in self.sch_config.keys()
        if self.sch_config['type'] == 'cos_decay_with_warmup':
            assert 't_max', 't_warmup' in self.scheduler_config.keys()
            scheduler = CosineAnnealingLRWarmup(
                optimizer,
                T_max=self.sch_config['t_max'],
                T_warmup=self.sch_config['t_warmup']
            )
        elif self.sch_config['type'] == 'multistep':
            assert 'milestones', 'gamma' in self.scheduler_config.keys()
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.sch_config['milestones'],
                gamma=self.sch_config['gamma']
            )
        else:
            print(f"Unknown schedular type {self.sch_config['type']}")
            assert False

        return [optimizer], [scheduler]
