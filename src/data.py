import os, sys
sys.path.append(os.path.abspath('../src'))
import numpy as np
import h5py
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import HandlerClass
from modules.module_util import adjust_array_len, clip_segment


class MMDemos_small(Dataset):
    def __init__(
        self,
        path, handler,
        train_split=0.8,
        multiplier=2,
        seed=114514,
        seg_len=None,    # only use part of the whole demo./traj.
        init_mode='train'
    ):
        super().__init__()
        
        # first load data
        # need to think of other smarter ways when dataset is very large...
        self.load_data(path, handler)
        
        self.n_demos = len(self.DataList[0])
        # self.dump_data_example()
        print(f'MMDemos_small have {self.n_demos} demos.')
        
        # get training or eval split
        self.train_split = train_split
        self.multiplier = multiplier
        self.n_train = int(self.n_demos * train_split)
        self.n_train_multiplier = self.n_train * multiplier
        self.n_eval = self.n_demos - self.n_train
        ### randomly select based on seed
        self.seed = seed
        np.random.seed(seed)
        ### select a set of index to be train
        self.train_idx = np.random.choice(self.n_demos, self.n_train, replace=False)
        self.train_idx_multiplier = np.tile(self.train_idx, multiplier)
        self.eval_idx = np.setdiff1d(np.arange(self.n_demos), self.train_idx)
        self.dump_split()
        
        self.seg_len = seg_len if seg_len is not None else self.max_traj_len
        self.mode = init_mode
    
    
    # since we may have many different dataset...
    # we need a way to customize each of them...
    def load_data(self, path, handler):
        f_handler = getattr(HandlerClass, handler, None)
        if f_handler is None:
            raise ValueError(f"Function '{handler}' not found in the HandlerClass!")
        
        self.DataList, self.n_modal, self.max_traj_len, self.modal_dims, self.Optical = f_handler(path)
    
    # see whether we use train_idx_multiplier or eval_idx
    def set_mode(self, mode):
        assert mode in ["train", "eval"]
        self.mode = mode
        
    
    def __len__(self):
        if self.mode == 'train':
            return self.n_train_multiplier
        elif self.mode == 'eval':
            return self.n_eval
        else:
            print(f'Unknown MMDemos_small mode: {self.mode}')
            assert False
    
    
    def __getitem__(self, index):
        if self.mode == 'train':
            idx = self.train_idx_multiplier[index]
            in_list = [self.DataList[i_m][idx]['data'].astype(np.float32) for i_m in range(self.n_modal)]
            clipped_in_list, t = adjust_array_len(in_list, self.seg_len)
            
            # optic = self.Optical[idx]['data'][self.optic].astype(np.float32)
            # clipped_optic = clip_segment(optic, t, self.seg_len, 0.0)
            if isinstance(self.Optical, int):
                data_dict = {'t': np.array([t]).astype(np.int32),
                             'task_id': np.array([self.DataList[0][idx]['task_id'] / self.Optical]).astype(np.float32),
                             'lengths': np.array(self.seg_len).astype(np.int32),
                             'in': clipped_in_list}
            else:
                data_dict = {'t': np.array([t]).astype(np.int32),
                             'lengths': np.array(self.seg_len).astype(np.int32),
                             'in': clipped_in_list}
        
        elif self.mode == 'eval':
            idx = self.eval_idx[index]
            in_list = [self.DataList[i_m][idx]['data'].astype(np.float32) for i_m in range(self.n_modal)]
            clipped_in_list, t = adjust_array_len(in_list)
            data_dict = {'t': np.array([t]).astype(np.int32),
                         'lengths': np.array(self.seg_len).astype(np.int32),
                         'in': clipped_in_list}
        
        else:
            print(f"Unknown self.mode of MMDemos_small: {self.mode}")
            assert False
        
        return data_dict
    
        
    def dump_data_example(self):
        for modallist in self.DataList:
            for i in range(10):
                for k in modallist[i].keys():
                    if k != 'data':
                        print(modallist[i][k])
                    else:
                        print(modallist[i][k].shape)
                print('-'*100)
                
    def dump_split(self):
        print(f'type(self.train_idx): {type(self.train_idx)}\tlen(self.train_idx): {len(self.train_idx)}')
        print(f'self.train_idx: {self.train_idx}')
        print(f'type(self.train_idx_multiplier): {type(self.train_idx_multiplier)}\tlen(self.train_idx_multiplier): {len(self.train_idx_multiplier)}')
        print(f'self.train_idx_multiplier: {self.train_idx_multiplier}')
        print(f'type(self.eval_idx): {type(self.eval_idx)}\tlen(self.eval_idx): {len(self.eval_idx)}')
        print(f'self.eval_idx: {self.eval_idx}')


# To obtain the padding function for sequences.
def get_padding_fn(data_names):
    assert 's' in data_names, 'Should at least include `s` in data_names.'

    def pad_collate(*args):
        assert len(args) == 1
        output = {k: [] for k in data_names}
        for b in args[0]:  # Batches
            for k in data_names:
                output[k].append(torch.from_numpy(b[k]))

        # Include the actual length of each sequence sampled from a trajectory.
        # If we set max_seq_length=min_seq_length, this is a constant across samples.
        output['lengths'] = torch.tensor([len(s) for s in output['s']])

        # Padding all the sequences.
        for k in data_names:
            output[k] = pad_sequence(output[k], batch_first=True, padding_value=0)

        return output

    return pad_collate
    

# Sample code for the data loader.
if __name__ == "__main__":
    
    train_dataset = MMDemos_small(
        path="/group/ycyang/rzliu/datasets/raw/bridge_data_v2/deepthought_folding_table",
        handler="read_scripted_raw",
        train_split=1.0,
        multiplier=2,
        seed=2333,
        seg_len=20,
        init_mode='train'
    )
    
    from torch.utils.data import DataLoader
    
    trainloader = DataLoader(
        dataset=train_dataset, 
        batch_size=4,  # batch_size,
        shuffle=True
    )
    
    iter = iter(trainloader)
    while True:
        data = next(iter) 
        for k, v in data.items():
            if k == 't' and torch.any(v):
                print(k, v)
            if k == 'in':
                print(k, v[0].shape, v[1].shape)
