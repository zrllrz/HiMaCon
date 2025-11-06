import os, sys, shutil
import numpy as np
import math
import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt


def arccos_hinge(vcos, hinge):
    # vcos: [B, T]
    # hinge: [B, T] allow each have a different value
    return torch.maximum(0, hinge - torch.arccos(vcos))


def copy_file(src_file, dest_dir, new_filename=None):
    try:
        # Ensure the destination directory exists
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # If a new filename is provided, set the destination path with the new name
        if new_filename:
            dest_file = os.path.join(dest_dir, new_filename)
        else:
            # Keep the original filename if no new name is provided
            dest_file = os.path.join(dest_dir, os.path.basename(src_file))
        
        # Perform the file copy operation
        shutil.copy(src_file, dest_file)
        print(f"File '{src_file}' successfully copied to '{dest_file}'")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_functions(functions_dict, num_points=100):
    """
    Plot a set of functions from [0, 1] -> R on separate subplots.
    
    Parameters:
    - functions_dict (dict): Dictionary where keys are function names (str), and values are callable functions.
    - num_points (int): Number of points to plot for each function. Default is 100.
    """
    num_functions = len(functions_dict)
    
    # Create a figure with subplots, one for each function
    fig, axes = plt.subplots(num_functions, 1, figsize=(6, 4 * num_functions))
    
    # If only one function, axes is not a list, make it a list
    if num_functions == 1:
        axes = [axes]
    
    # Define the domain [0, 1] with 'num_points' points
    x = np.linspace(0, 1, num_points)
    
    # Plot each function in its own subplot
    for i, (name, func) in enumerate(functions_dict.items()):
        y = func(x)  # Evaluate the function at each point in x
        print(x, '\n', y)
        axes[i].plot(x, y, label=name)
        axes[i].set_title(f"Function: {name}")
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')
        axes[i].legend()
    
    plt.tight_layout()
    
    plt.savefig("tmp_func_display.jpg")
    print(f"Figure saved as tmp_func_display.jpg")
    
    # Close the plot to free memory
    plt.close()



# Safe eval function to return a handler (callable function)
def safe_handler(expression):
    # Define a function that evaluates the expression dynamically for a given value of 't'
    def handler(t):
        # Allowed variables/functions for evaluation
        allowed_vars = {
            't': t,
            'sin': np.sin,
            'cos': np.cos,
            'log': np.log,
            'exp': np.exp,
            'abs': np.abs,
            'where': np.where
            # Add other safe math functions here if needed
        }
        
        try:
            if 't' not in expression:
                # First, try evaluating the expression without 't' to check if it's a constant
                constant_value = eval(expression, {"__builtins__": None}, {})
                # If eval succeeds without using 't', we treat it as a constant function
                return np.full_like(t, constant_value)
            else:
                # the expression depends on 't', evaluate normally
                return eval(expression, {"__builtins__": None}, allowed_vars)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {expression}. Error: {e}")
        
        # try:
        #     return eval(expression, {"__builtins__": None}, allowed_vars)
        # except Exception as e:
        #     raise ValueError(f"Error evaluating expression: {expression}. Error: {e}")
    
    # Return the handler function
    return handler


def anomaly_score(r, sharpen=100.0):
    f = torch.abs(r - 1.0)
    f = ((r - 1.0) + f) * 0.5
    return (f / r) ** (1.0 / sharpen)

def cos_anomaly_score(vcos, sharpen=10.0):
    f = vcos * sharpen
    f = (torch.abs(torch.tanh(f)) - torch.tanh(f)) * 0.5
    return f


def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses), losses
    else:
        assert losses.shape == weights.shape, losses.shape
        return torch.mean(losses * weights), losses * weights


def l1_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean(torch.abs(preds - targets), dim=-1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape,f"weights shape {weights.shape} is not the same. Should be {losses.shape}"
        return torch.div(torch.sum(losses * weights), torch.sum(weights)+1e-12)

def l2_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets)**2, dim=-1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape,f"weights shape {weights.shape} is not the same. Should be {losses.shape}"
        return torch.div(torch.sum(losses * weights), torch.sum(weights)+1e-12)


def get_norm_loss(preds: torch.Tensor, targets, lengths, mode=1):
    B = preds.shape[0]
    # print(lengths)
    # print(f"lengths.shape = {lengths.shape}")
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    # print(f"max_len = {max_len}")
    lengths = lengths[:, None]  # B x 1
    # print(f"lengths.shape = {lengths.shape}")
    temp = torch.arange(0, max_len)[None].expand(B, -1).cuda()  # B x max_len
    # print(f"temp.shape = {temp.shape}")
    # print(f"lengths.expand(B, max_len) = {lengths.expand(B, max_len)}")
    masks = torch.less(temp, lengths.expand(B, max_len)).float()  # B x max_len
    # (temp < lengths.expand(B, max_len)).float()  # B x max_len
    if mode == 1:
        return l1_loss_with_weights(preds, targets, masks)
        # return l1_loss_with_weights(preds.reshape(-1, preds.size(-1)),
        #                             targets.reshape(-1, targets.size(-1)),
        #                             masks.reshape(-1))
    elif mode == 2:
        return l2_loss_with_weights(preds, targets, masks)
        # return l1_loss_with_weights(preds.reshape(-1, preds.size(-1)),
        #                             targets.reshape(-1, targets.size(-1)),
        #                             masks.reshape(-1))
    else:
        print(f"Unknown loss calculation mode: {mode, type(mode)}")
        assert False


def get_loss(preds, targets, lengths):
    # If we have sequences of varied lengths, use masks so we do not compute loss
    # over padded values. If we set max_seq_length=min_seq_length, then it should
    # not matter since all sequences have the same length.
    B = preds.shape[0]
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    lengths = lengths[:, None]  # B x 1
    temp = torch.arange(0, max_len)[None].expand(B, -1).cuda()  # B x max_len
    masks = torch.less(temp, lengths.expand(B, max_len)).float()  # B x max_len
    # (temp < lengths.expand(B, max_len)).float()  # B x max_len

    loss, lossess = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)),
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss, lossess.reshape(B, -1)


def init_centroids(datas, n_centroids):
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    cent_init = datas[i0][None, :]
    for _ in range(n_centroids - 1):
        d = torch.sum(datas ** 2, dim=1, keepdim=True) + \
            torch.sum(cent_init ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', datas, rearrange(cent_init, 'n d -> d n'))

        d_min = d.min(dim=1)[0] + 1e-5
        d_min = torch.where(torch.isnan(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.less(d_min, 0.0), torch.zeros_like(d_min), d_min)
        d_min_sum = d_min.sum()

        if d_min_sum == 0.0 or torch.isnan(d_min_sum) or torch.isinf(d_min_sum):
            i = torch.randint(0, N, (1,))[0]
        else:
            i = d_min.multinomial(num_samples=1)[0]

        cent_init = torch.cat([cent_init, datas[i][None, :]], dim=0)
    return cent_init


def init_centroids_neighbor(datas, unified_t, n_centroids):
    # datas: (B*T, e_dim), key_soft
    # unified_t: (B*T), unified timesteps
    # n_centroids: number of centroids
    N, D = datas.shape
    i0 = torch.randint(0, N, (1,))[0]
    unified_t_unsq = unified_t.view(-1, 1)
    cent_init_ind = torch.tensor([i0])
    cent_init_u_t = unified_t[i0].view(1)
    for _ in range(n_centroids - 1):
        d = torch.abs(unified_t_unsq - cent_init_u_t)
        d_min = d.min(dim=1)[0] + 1e-5
        d_min = torch.where(torch.isnan(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
        d_min = torch.where(torch.less(d_min, 0.0), torch.zeros_like(d_min), d_min)
        d_min_sum = d_min.sum()

        if d_min_sum == 0.0 or torch.isnan(d_min_sum) or torch.isinf(d_min_sum):
            i = torch.randint(0, N, (1,))[0]
        else:
            i = d_min.multinomial(num_samples=1)[0]
        cent_init_ind = torch.cat([cent_init_ind, torch.tensor([i])], dim=0)
        cent_init_u_t = torch.cat([cent_init_u_t, unified_t_unsq[i]], dim=0)


    # sorted by unified time step!!!!!!
    _, sorted_sub_ind = torch.sort(cent_init_u_t)
    cent_init_ind = cent_init_ind[sorted_sub_ind.to(cent_init_ind.device)]
    cent_init_datas = datas[cent_init_ind]

    return cent_init_datas
