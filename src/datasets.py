'''
Read data from different Demo. Datasets
into same format
We also hope that your dataset put the information of an episode (trajectory)
into a same path, instead of separate different information types
E.g.
Dataset
-- traj_0
---- img
------ im_0.jpg
------ im_1.jpg
------ ...
------ im_{T-1}.jpg
---- action.npy
---- ...
-- traj_1
-- ...
-- traj_{N-1}

Always return:

# a list of multi-modality data
[
    (modal_0:) [
        {
            data_path (str): parent path of the original demos, 
            data ([T, ...]): data in tensor, length L, arbitrary shape at each time-step
        }
    ]
    (modal_1:)
    ...
    (modal_{m-1}:)
]

'''

import os
from pathlib import Path
import pickle
import numpy as np
import h5py
from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

from tqdm import tqdm
from time import sleep


def create_gif_from_numpy(array, gif_path, duration=100):
    # Ensure the array has the correct shape [T, 128, 128, 3]
    assert array.ndim == 4 and array.shape[-1] == 3, "Input shape must be [T, 128, 128, 3]."
    
    # Normalize values to the range [0, 255]
    array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

    # Convert frames to a list of PIL Images
    frames = [Image.fromarray(array[t], 'RGB') for t in range(array.shape[0])]

    # Save as GIF
    frames[0].save(
        gif_path, 
        save_all=True, 
        append_images=frames[1:], 
        duration=duration, 
        loop=0
    )
    print(f"GIF saved at {gif_path}")


# Customized reading scripted_raw of BridgeDataV2
# Notice that some traj have differen views, some only have one views
# So we will instead use the "images[num]" as the "traj"
class HandlerClass:
    def read_aloha_500_50_human(aloha_path, print_dump=False):
        DataList = []
        DataList0 = []
        DataList1 = []
        max_traj_len = 0
        max_traj_path = ""
        
        episodes = [epi for epi in os.listdir(aloha_path) if epi.startswith("episode_")]
        
        for epi in episodes:
            compress_cam_high = np.load(f"{aloha_path}/{epi}/compress-cam_high.npy")[4::10,:]
            traj_len = compress_cam_high.shape[0]
            max_traj_len = max(max_traj_len, traj_len)
            max_traj_path = f"{aloha_path}/{epi}" if traj_len == max_traj_len else max_traj_path
                    
            compress_depth_map = np.load(f"{aloha_path}/{epi}/compress-depth_map.npy")[4::10,:]
            
            DataList0.append(
                {'data_path': f"{aloha_path}/{epi}",
                 'type': 'img', 
                 'data': compress_cam_high})
            DataList1.append(
                {'type': 'img', 
                 'data': compress_depth_map})

            if print_dump:
                for img, pose in zip(DataList0, DataList1):
                    print(
                        f"data_path: {img['data_path']}\n",
                        f"type: {img['type']}, size: {img['data'].shape}\n",
                        f"type: {pose['type']}, size: {pose['data'].shape}\n"
                    )
        
        # some statistics
        print(f"Get a dataset of {len(DataList0)} items\nMax traj. len is {max_traj_len} of {max_traj_path}")
        DataList=[DataList0, DataList1]
        modal_dims = [modal[0]['data'].shape[-1] for modal in DataList]
        return DataList, 2, max_traj_len, modal_dims, None
    
    
    def read_aloha_500_50(aloha_path, print_dump=False):
        DataList = []
        DataList0 = []
        DataList1 = []
        DataList2 = []
        DataList3 = []
        max_traj_len = 0
        max_traj_path = ""
        
        episodes = [epi for epi in os.listdir(aloha_path) if epi.startswith("episode_")]
        
        for epi in episodes:
            compress_cam_high = np.load(f"{aloha_path}/{epi}/compress-cam_high.npy")[4::10,:]
            traj_len = compress_cam_high.shape[0]
            max_traj_len = max(max_traj_len, traj_len)
            max_traj_path = f"{aloha_path}/{epi}" if traj_len == max_traj_len else max_traj_path
                    
            compress_cam_left_wrist = np.load(f"{aloha_path}/{epi}/compress-cam_left_wrist.npy")[4::10,:]
            compress_cam_right_wrist = np.load(f"{aloha_path}/{epi}/compress-cam_left_wrist.npy")[4::10,:]
            prop = np.load(f"{aloha_path}/{epi}/prop.npy")[4::10,:]
            
            DataList0.append(
                {'data_path': f"{aloha_path}/{epi}",
                 'type': 'img', 
                 'data': compress_cam_high})
            DataList1.append(
                {'type': 'img', 
                 'data': compress_cam_left_wrist})
            DataList2.append(
                {'type': 'img', 
                 'data': compress_cam_right_wrist})
            DataList3.append(
                {'type': 'pose', 
                 'data': prop})

            if print_dump:
                for img, pose in zip(DataList0, DataList1, DataList2, DataList3):
                    print(
                        f"data_path: {img['data_path']}\n",
                        f"type: {img['type']}, size: {img['data'].shape}\n",
                        f"type: {pose['type']}, size: {pose['data'].shape}\n"
                    )
        
        # some statistics
        print(f"Get a dataset of {len(DataList0)} items\nMax traj. len is {max_traj_len} of {max_traj_path}")
        DataList=[DataList0, DataList1, DataList2, DataList3]
        modal_dims = [modal[0]['data'].shape[-1] for modal in DataList]
        return DataList, 4, max_traj_len, modal_dims, None
        
    
    def read_libero(libero_path, print_dump=False):
        DataList = []
        DataList0 = []
        DataList1 = []
        DataList2 = []
        Optical = []
        max_traj_len = 0
        max_traj_path = ""
        
        opt_names = ['norm_agentview_delta_F_m.npy', 'norm_eye_in_hand_delta_F_m.npy']
        
        walked_libero_path = os.walk(libero_path)
        
        task_id_map = {}
        task_id_max = 0
        
        for dirpath, dirnames, filenames in tqdm(walked_libero_path):
            if print_dump:
                print(f"Directory path: {dirpath}")
                print(f"Subdirectories: {dirnames}")
                print(f"Files: {filenames}")
                print('-' * 40)
            
            for dirname in dirnames:
                if dirname[:5] == 'demo_':
                    compress_agentview_rgb = np.load(f'{dirpath}/{dirname}/compress_agentview_rgb.npy')
                    compress_eye_in_hand_rgb = np.load(f'{dirpath}/{dirname}/compress_eye_in_hand_rgb.npy')
                    robot_states = np.load(f'{dirpath}/{dirname}/robot_states.npy')
                    
                    traj_len = compress_agentview_rgb.shape[0]
                    max_traj_len = max(max_traj_len, traj_len)
                    max_traj_path = f"{dirpath}/{dirname}" if traj_len == max_traj_len else max_traj_path
                    
                    task_name = dirpath.split('/')[-1]
                    if task_name not in task_id_map.keys():
                        task_id_map[task_name] = task_id_max
                        task_id_max += 1
                    task_id_now = task_id_map[task_name]
                    
                    DataList0.append(
                        {'data_path': f'{dirpath}/{dirname}',
                         'task_id': task_id_now,
                         'type': 'agentview_rgb', 
                         'data': compress_agentview_rgb})
                    DataList1.append(
                        {'type': 'eye_in_hand_rgb',
                         'data': compress_eye_in_hand_rgb})
                    DataList2.append(
                        {'type': 'robot_states',
                         'data': robot_states })
                    
                    if print_dump:
                        for img1, img2, pose, optic in zip(DataList0, DataList1, DataList2, Optical):
                            print(
                                f"data_path: {img1['data_path']}\n",
                                f"type: {img1['type']}, size: {img1['data'].shape}\n",
                                f"type: {img2['type']}, size: {img2['data'].shape}\n",
                                f"type: {pose['type']}, size: {pose['data'].shape}\n",
                                f"type: {optic['type']}, {optic['data']}\n",
                            )
                    if print_dump:
                        input()
                    
        # some statistics
        print(f"Get a dataset of {len(DataList0)} items\nMax traj. len is {max_traj_len} of {max_traj_path}")
        DataList=[DataList0, DataList1, DataList2]
        modal_dims = [modal[0]['data'].shape[-1] for modal in DataList]
        print("modal_dims:", modal_dims)
        print("task_id_total:", task_id_max)
        
        return DataList, 3, max_traj_len, modal_dims, task_id_max
    
    def read_scripted_raw(scripted_raw_path, print_dump=False):
        # collect all
        DataList = []
        DataList0 = []
        DataList1 = []
        max_traj_len = 0
        max_traj_path = ""
        
        # currently we use 2 modalities
        # [(img), (7DoF pose)]

        # Walk through the directory tree
        walked_scripted_raw_path = os.walk(scripted_raw_path)
        for dirpath, dirnames, filenames in tqdm(walked_scripted_raw_path):
            if print_dump:
                print(f"Directory path: {dirpath}")
                print(f"Subdirectories: {dirnames}")
                print(f"Files: {filenames}")
                print('-' * 40)
        
            for dirname in dirnames:
                if dirname == "images0":
                    compress_img = np.load(f"{dirpath}/{dirname}/compress-images0.npy")
                    traj_len = compress_img.shape[0]
                    max_traj_len = max(max_traj_len, traj_len)
                    max_traj_path = f"{dirpath}/{dirname}" if traj_len == max_traj_len else max_traj_path
                    # pose
                    pose_path = f'{dirpath}/obs_dict.pkl'
                    if os.path.exists(pose_path) and os.path.isfile(pose_path):
                        with open(pose_path, 'rb') as fpkl:
                            pose = pickle.load(fpkl)['full_state']
                    else:
                        continue
                
                    # Now we are sure no modality is missing, create the data item in memory
                    DataList0.append(
                        {'data_path': f'{dirpath}/{dirname}',
                         'type': 'img', 
                         'data': compress_img})
                    DataList1.append(
                        {'type': 'pose',
                         'data': pose})
                
                    if print_dump:
                        for img, pose in zip(DataList0, DataList1):
                            print(
                                f"data_path: {img['data_path']}\n",
                                f"type: {img['type']}, size: {img['data'].shape}\n",
                                f"type: {pose['type']}, size: {pose['data'].shape}\n"
                            )
                    if print_dump:
                        input()
            
        # some statistics
        print(f"Get a dataset of {len(DataList0)} items\nMax traj. len is {max_traj_len} of {max_traj_path}")
        DataList=[DataList0, DataList1]
        modal_dims = [modal[0]['data'].shape[-1] for modal in DataList]
        return DataList, 2, max_traj_len, modal_dims, None


if __name__ == '__main__':
    HandlerClass.read_aloha_500_50(f"/group/ycyang/rzliu/MInfoCon/dataset_aloha/sunli/clean_cup_easy_2", False)
