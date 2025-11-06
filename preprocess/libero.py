"""
Process original LIBERO dataset to the dataset we use
"""

import os, re, h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers import StableDiffusionPipeline
import torchvision.transforms as T

from tqdm import tqdm


# Function to preprocess the input image
def preprocess_image(image_path, image_size=(128, 128)):
    """
    Preprocess the image to match the input size for the VAE encoder.
    
    Args:
    - image_path (str): Path to the input image.
    - image_size (tuple): Desired size of the image (default is 512x512 for Stable Diffusion).
    
    Returns:
    - torch.Tensor: Preprocessed image as a PyTorch tensor.
    """
    image = Image.open(image_path).convert("RGB")
    preprocess = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to("cuda")  # Add batch dimension and move to GPU
    return image_tensor


# Function to preprocess the input numpy
def preprocess_image_npy(image, image_size=(128, 128)):
    """
    Preprocess the image to match the input size for the VAE encoder.
    
    Args:
    - image_path (str): Path to the input image.
    - image_size (tuple): Desired size of the image (default is 512x512 for Stable Diffusion).
    
    Returns:
    - torch.Tensor: Preprocessed image as a PyTorch tensor.
    """
    image = np.ascontiguousarray(image)
    image_th = torch.tensor(image)
    image_th = image_th.cuda().to(dtype=torch.float32)
    image_th = image_th.permute(0, 3, 1, 2)
    image_th = image_th / 127.5 - 1.0
    # image_th = image_th / 255.0
    preprocess = T.Compose([
        T.Resize(image_size),
    ])
    image_tensor = preprocess(image_th).to("cuda")  # Add batch dimension and move to GPU
    return image_tensor


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libero_org_path', type=str, help="path of original LIBERO dataset")
    parser.add_argument('--libero_tar_path', type=str, help="path to save processed LIBERO dataset")
    parser.add_argument('--chunk_size', type=int, help="chunk size when processing images")
    args = parser.parse_args()
    print(args)
    
    
    libero_path = args.libero_org_path
    walked_libero_path = os.walk(libero_path)
    
    # Load the pre-trained Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to("cuda")
    # You only need the VAE encoder part for encoding the image
    vae_part = pipe.vae
    
    n_demo_total = 0
    for dirpath, dirnames, filenames in walked_libero_path:
        for filename in filenames:
            if filename.endswith('.hdf5'):
                print(f'{dirpath}/{filename}')
                task_description = os.path.basename(f'{dirpath}/{filename}')
                task_parent_path = os.path.dirname(f'{dirpath}/{filename}')
                task_suite_name = os.path.basename(task_parent_path)
                print(task_suite_name, ":", task_description)
                target_path = f"{args.libero_tar_path}/{task_suite_name}/{task_description}"[:-len(".hdf5")]
                print(target_path)
                os.makedirs(f"{args.libero_tar_path}/{task_suite_name}", exist_ok=True)
                os.makedirs(target_path, exist_ok=True)
                
                with h5py.File(f'{dirpath}/{filename}', 'r') as f:
                    f_keys = list(f.keys())
                    assert f_keys == ['data'],f"\n{dirpath}/{filename}.keys()\n{f_keys}"
                    
                    n_traj = len(list(f['data'].keys()))
                    n_demo_total += n_traj
                    f_data_keys = list(f['data'].keys())
                    for d in f_data_keys:
                        assert d in [f'demo_{ix}' for ix in range(n_traj)],f"\n{dirpath}/{filename}[\'data\'].keys()\n{f_data_keys}"
                    print(f'num of demos: {n_traj}')
                    
                    for i_demo in tqdm(range(n_traj)):
                        demo_path = f'{target_path}/demo_{i_demo}'
                        os.makedirs(demo_path, exist_ok=True)
                        
                        for item in ['actions', 'dones', 'obs', 'rewards', 'robot_states', 'states']:
                            np.save(f'{demo_path}/{item}.npy', np.ascontiguousarray(np.array(f['data'][f'demo_{i_demo}'][f'{item}'])))
                        
                        for obs_item in ['ee_ori', 'ee_pos', 'ee_states', 'gripper_states', 'joint_states']:
                            np.save(f'{demo_path}/{obs_item}.npy', np.ascontiguousarray(np.array(f['data'][f'demo_{i_demo}']['obs'][f'{obs_item}'])))
                        
                        agentview_rgb = np.array(f['data'][f'demo_{i_demo}']['obs']['agentview_rgb'])[:, ::-1, :, :]
                        assert len(agentview_rgb.shape) == 4 and list(agentview_rgb.shape[1:]) == [128,128,3],f"\n{dirpath}/{filename}[\'data\'][\'demo_{i_demo}\'][\'obs\'][\'agentview_rgb\'].shape\n{agentview_rgb.shape}"
                        
                        eye_in_hand_rgb = np.array(f['data'][f'demo_{i_demo}']['obs']['eye_in_hand_rgb'])
                        assert len(eye_in_hand_rgb.shape) == 4 and list(eye_in_hand_rgb.shape[1:]) == [128,128,3],f"\n{dirpath}/{filename}[\'data\'][\'demo_{i_demo}\'][\'obs\'][\'eye_in_hand_rgb\'].shape\n{eye_in_hand_rgb.shape}"
                        
                        with torch.no_grad():
                            def compress_image(image_npy, chunk_size=64):
                                N_image = image_npy.shape[0]
                                compress_img_list = []
                                for i in range(0, N_image, chunk_size):
                                    chunk_npy = image_npy[i:min(i+chunk_size,N_image)]
                                    chunk_tensor = preprocess_image_npy(chunk_npy)
                                    latent_representation = vae_part.encode(chunk_tensor).latent_dist.sample() * 0.18215
                                    compress_img_list.append(latent_representation)
                                compress_img = torch.cat(compress_img_list, dim=0).view(N_image, -1)
                                return compress_img
                            
                            compress_agentview_rgb = compress_image(agentview_rgb)
                            compress_eye_in_hand_rgb = compress_image(eye_in_hand_rgb)
                            assert compress_agentview_rgb.shape[0] == agentview_rgb.shape[0]
                            assert compress_eye_in_hand_rgb.shape[0] == agentview_rgb.shape[0]
                            np.save(f'{demo_path}/compress_agentview_rgb.npy', compress_agentview_rgb.detach().cpu().numpy())
                            np.save(f'{demo_path}/compress_eye_in_hand_rgb.npy', compress_eye_in_hand_rgb.detach().cpu().numpy())
    
    print(f"processed {n_demo_total} demos")
