"""
Reference:
https://github.com/ashawkey/stable-dreamfusion
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def save_frequency_plot(data_counts, file_name='frequency_plot.jpg'):
    """
    Generates and saves a bar plot for the frequency of data items.
    
    Parameters:
    - data_counts (list or array): A list or array of frequencies [n1, n2, ..., nN].
    - file_name (str): The file name to save the plot as (default is 'frequency_plot.jpg').
    """
    N = len(data_counts)
    y = np.arange(N)  # X-axis: item indices
    
    # Create the bar plot
    plt.figure(figsize=(5, N*0.1))  # Scale the figure width with respect to N
    plt.barh(y, data_counts, height=0.1)
    
    # Set labels and title
    plt.ylabel('Items')
    plt.xlabel('Frequency')
    plt.title('Frequency of Data Items')
    
    # Set y-axis to be the same for all figures
    plt.xlim(0, max(data_counts) * 1.1)  # 10% extra space above the max count
    
    # Save the plot to a .jpg file
    plt.savefig(file_name, format='jpg')
    plt.close()


def normalized_pca(data: torch.Tensor, component: int):
    square_data = torch.div(torch.matmul(data.T, data), data.shape[0])
    U, S, Vh = torch.linalg.svd(square_data)
    U_k = U[:, :component]
    res = torch.matmul(data, U_k)
    return res


def proto_specific_visualize(data_tensor: torch.Tensor, prototypes_tensor: torch.Tensor, assignment: torch.Tensor):
    # data_tensor [N, d]
    # prototypes_tensor [K, d]
    # assignment [N] [0,1,...,(K-1)]
    K = prototypes_tensor.shape[0]
    cos_sim = torch.mm(prototypes_tensor, data_tensor.T)  # (K, N)
    
    # position in the chart
    arange = torch.arange(K, device=data_tensor.device)
    assign_map = (arange.unsqueeze(1) == assignment.unsqueeze(0)).to(dtype=torch.float32)
    # for each row (0~(K-1)), 1 means this data is assigned with row-idx concept, 0 otherwise
    # So we are able to assigne them to different lines
    assign_y = arange.unsqueeze(1) + 0.5 * assign_map
    # for concept row-idx k
    # data belongs to k will lie at [ y = k + 1/2 ]
    # data otherwise will lie at [ y = k]
    # we are able to see "discrimination ability" of different concepts.
    
    # now we can try to draw it
    x_points = cos_sim.view(-1).detach().cpu().numpy()
    y_points = assign_y.view(-1).detach().cpu().numpy()
    
    plt.figure(figsize=(6, (K+1)*2))
    
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_points, y_points, color='blue', label='Points')
    
    for k in range(K):
        plt.axhline(y=k, color='green', linestyle='-', linewidth=1, label=f'concept {k}')  # Avoid repeating labels
        plt.text(-1.2, k - 0.15, f"concept not {k}", verticalalignment='center', color='orange', fontsize=24)  # Label for y = k
        
        plt.axhline(y=k + 0.5, color='red', linestyle='-', linewidth=1, label=f'concept not {k}')
        plt.text(-1.2, k + 0.5 - 0.15, f"concept {k}", verticalalignment='center', color='orange', fontsize=24)  # Label for y = k + 1/2

    # plt.legend()
    plt.grid(False)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-0.5, K)
    plt.xticks(torch.arange(-1.0, 1.1, step=0.5))  # Show ticks at intervals of 0.5 on the x-axis
    plt.yticks(torch.arange(0.0, -0.4 + K, step=0.5))  # Show ticks at intervals of 0.5 on the y-axis
    # plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='datalim')
    # ax = plt.gca()  # Get current axis
    # ax.spines['left'].set_position(('data', -1.5))  # Fix y-axis at x=-1.5
    # ax.spines['bottom'].set_position(('data', -0.5))  # Fix x-axis at y=-0.5
    
    
    plt.savefig(f"discrimination_vis.jpg")
    plt.close()


def pca_visualize_prototypes(data_tensor: torch.Tensor, prototypes_tensor: torch.Tensor):
    """
    Visualizes data vectors and their closest prototypes based on cosine similarity.
    
    Parameters:
    - data_tensor: A tensor of shape [B, D] containing B data vectors.
    - prototypes_tensor: A tensor of shape [K, D] containing K prototype vectors.
    """
    # Step 1: Normalize the tensors for cosine similarity
    data_normalized = F.normalize(data_tensor, p=2, dim=1)
    prototypes_normalized = F.normalize(prototypes_tensor, p=2, dim=1)

    # Step 2: Calculate cosine similarity using PyTorch
    cosine_similarities = torch.matmul(data_normalized, prototypes_normalized.T)  # Shape [B, K]

    # Step 3: Find the closest prototype for each vector
    closest_prototypes = torch.argmax(cosine_similarities, dim=1)  # Shape [B]

    # Step 4: Combine original vectors and prototypes for PCA
    combined_data = torch.cat((data_normalized, prototypes_normalized), dim=0)  # Shape [B + K, D]

    # Step 5: Perform PCA using sklearn
    # pca = PCA(n_components=2)
    reduced_data = normalized_pca(data=combined_data, component=2).detach().cpu().numpy()

    # Step 6: Visualize the results
    plt.figure(figsize=(10, 6))
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)
    plt.gca().set_aspect('equal', adjustable='box')

    # Define colors for prototypes
    num_prototypes = prototypes_tensor.size(0)
    colors = plt.cm.get_cmap('tab10', num_prototypes)

    # Plot original data points with colors based on closest prototype
    for prototype_idx in range(num_prototypes):
        indices = (closest_prototypes == prototype_idx).nonzero(as_tuple=True)[0].cpu().numpy()
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
                    color=colors(prototype_idx))
        # Plot prototypes with larger markers
        plt.scatter(reduced_data[data_tensor.size(0) + prototype_idx, 0],
                    reduced_data[data_tensor.size(0) + prototype_idx, 1],
                    color=colors(prototype_idx), marker='x', s=200) # label=f'Prototype {prototype_idx} (Center)')
    
    plt.title('PCA Visualization of Codebook')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.legend()
    plt.grid()
    plt.savefig(f"code_book_vis.jpg")
    plt.close()


def tsne_visualize_prototypes(data_tensor: torch.Tensor, prototypes_tensor: torch.Tensor):
    """
    Visualizes data vectors and their closest prototypes based on cosine similarity using t-SNE.
    
    Parameters:
    - data_tensor: A tensor of shape [B, D] containing B data vectors.
    - prototypes_tensor: A tensor of shape [K, D] containing K prototype vectors.
    """
    # Step 1: Normalize the tensors for cosine similarity
    data_normalized = torch.nn.functional.normalize(data_tensor, p=2, dim=1)
    prototypes_normalized = torch.nn.functional.normalize(prototypes_tensor, p=2, dim=1)

    # Step 2: Calculate cosine similarity
    cosine_similarities = torch.mm(data_normalized, prototypes_normalized.T)  # Shape [B, K]

    # Step 3: Find the closest prototype for each vector
    closest_prototypes = torch.argmax(cosine_similarities, dim=1)  # Shape [B]

    # Step 4: Combine original vectors and prototypes
    combined_data = torch.cat((data_normalized, prototypes_normalized), dim=0)  # Shape [B + K, D]

    # Step 5: Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(combined_data.detach().cpu().numpy())  # Shape [B + K, 2]

    # Step 6: Visualize the results
    plt.figure(figsize=(10, 6))

    # Define a color map for different prototypes
    unique_prototypes = closest_prototypes.unique()
    colors = plt.cm.get_cmap('tab10', len(unique_prototypes))

    # Plot original data points with colors based on closest prototype
    for prototype_idx in unique_prototypes:
        indices = (closest_prototypes == prototype_idx).nonzero(as_tuple=True)[0]
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1],
                    color=colors(prototype_idx.item()), label=f'Prototype {prototype_idx.item()}', alpha=0.6)

    # Plot prototypes
    plt.scatter(reduced_data[data_tensor.size(0):, 0], reduced_data[data_tensor.size(0):, 1],
                c='black', marker='x', label='Prototypes', s=100)

    # Annotate the closest prototype for each data vector
    for i in range(data_tensor.size(0)):
        plt.annotate(f'{closest_prototypes[i].item()}', (reduced_data[i, 0], reduced_data[i, 1]), fontsize=8)

    plt.title('t-SNE Visualization of Vectors and Prototypes')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid()
    plt.show()


def adjust_array_len(input_list: list[np.ndarray], len: int) -> list[np.ndarray]:
    T = input_list[0].shape[0]
    
    if T < len:
        # Case 1: T <= len, we need to repeat the last slice [T-1:T, :]
        output_list = [
            np.concatenate([input, np.tile(input[-1:, :], (len - T, 1))])
            for input in input_list
        ]
        t = 0
    
    elif T > len:
        # Case 2: T > len, we randomly select a part of the input of shape [len, D]
        start = np.random.randint(0, T - len + 1)
        output_list = [
            input[start:start+len, :]
            for input in input_list
        ]
        t = start
    else:
        # Case 3: T == len, no modification needed
        output_list = input_list
        t = 0
    
    return output_list, t

def clip_segment(input: np.ndarray, t_begin: int, seg_len: int, padding=0.0) -> np.ndarray:
    T = input.shape[0]
    if t_begin + seg_len <= T:
        return input[t_begin:t_begin+seg_len]
    else:
        assert t_begin < T
        # padding
        if isinstance(padding, (int, float, complex)):
            left_shape = \
                tuple([t_begin + seg_len - T]) if len(input.shape) == 1 \
                else tuple([t_begin + seg_len - T] + list(input.shape[1:]))
            return np.concatenate([input[t_begin:], np.full(shape=left_shape, fill_value=padding)], axis=0)
        elif padding == 'last':
            left = np.stack([input[-1]] * (t_begin + seg_len - T), axis=0)
            return np.concatenate([input[t_begin:], left], axis=0)
        else:
            print(f'Unknown padding value: {padding}')
            assert False
        


def generate_non_zero_list(n):
    if n < 1:
        raise ValueError("Length of list must be at least 1")
    
    while True:
        sequence = [random.randint(0, 1) for _ in range(n)]
        if 0 in sequence:  # 只要不是全1就接受
            res = []
            for i in range(n):
                if sequence[i] == 0:
                    res.append(i)
            return res


def print_vector(vector, data="int"):
    """
    Print a vector in a formatted, multi-line manner.

    :param vector: List or 1D array to be printed.
    :param line_width: Number of elements per line.
    """
    if isinstance(vector, torch.Tensor):
        vector = vector.tolist()
        
    # Print the keys (column headers) in a single row
    keys_row = " | ".join(f"{i:<10}" for i in range(len(vector)))
    
    # Print a separator row
    separator_row = "-" * len(keys_row)
    
    if data == "int":
        # Print the values in a single row (right-aligned float values)
        values_row = " | ".join(f"{value:<10d}" for value in vector)
    else:
        assert data == "float"
        values_row = " | ".join(f"{value:>10.4f}" for value in vector)
    # Display the result
    print(keys_row)
    print(separator_row)
    print(values_row, '\n')


def print_dict(data_dict):
    """
    Print a dictionary with string keys and float values as a transposed row-based chart.
    
    :param data_dict: Dictionary where keys are strings and values are floats.
    """
    # Print the keys (column headers) in a single row
    keys_row = " | ".join(f"{key:<10}" for key in data_dict.keys())
    
    # Print a separator row
    separator_row = "-" * len(keys_row)
    
    # Print the values in a single row (right-aligned float values)
    values_row = " | ".join(f"{value:>10.4f}" for value in data_dict.values())
    
    # Display the result
    print(keys_row)
    print(separator_row)
    print(values_row, '\n')


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class unifiedTime(nn.Module):
    def __init__(self, n_e, eps=1e-12):
        super().__init__()
        self.n_e = n_e
        self.param_ut = nn.Parameter(torch.rand(n_e + 1, 1))
        self.eps = eps


    def forward(self):
        # no parameters!!!
        # we calculate the unified time according to parameters
        ut_cumsum = torch.cumsum(self.param_ut, dim=0)
        print(ut_cumsum)
        ut = torch.div(ut_cumsum, ut_cumsum[-1, 0] + self.eps)
        return ut[:-1]

    def resert_ut(self, u_t):
        with torch.no_grad():
            param_ut = torch.cat([u_t[0], u_t[1:] - u_t[:-1], 1.0 - u_t[-1]], dim=0)
            self.param_ut.data = param_ut


class FreqEncoder(nn.Module):
    def __init__(self, half_t_size, feature_size):
        super().__init__()
        self.feature_size = feature_size

        self.out_linear = nn.Linear(half_t_size * 2, feature_size)
        self.out_linear.weight.data.normal_(mean=0.0, std=0.02)
        self.out_linear.bias.data.zero_()

        coe_freq = torch.arange(half_t_size)
        coe_freq = torch.mul(torch.pow(2.0, coe_freq), torch.pi)
        self.register_buffer('coe_freq', coe_freq.unsqueeze(0))

    def forward(self, feature, unified_t):
        u_t = torch.mul(torch.sub(torch.mul(unified_t, 2.0), 1.0), torch.pi).unsqueeze(-1)
        emb_cos = torch.cos(u_t @ self.coe_freq)  # ()
        emb_sin = torch.sin(u_t @ self.coe_freq)
        emb_t = torch.cat([emb_cos, emb_sin], dim=-1)  # (..., 2 * half_t_size)
        emb_t = self.out_linear(emb_t)  # (..., feature_size)
        emb_t = emb_t + feature  # feature should be normalized...
        return emb_t


class TimeSphereEncoder(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate * 2.0

    def forward(self, feature, unified_t):
        # feature: (..., d)
        # unified_t: (...)
        u_t = torch.mul(torch.sub(torch.mul(unified_t, 2.0), 1.0), torch.pi).unsqueeze(-1)  # (..., 1)
        u_t_sin = torch.sin(torch.div(u_t, self.rate))  # (..., 1)
        u_t_cos = torch.cos(torch.div(u_t, self.rate))  # (..., 1)
        f_cos = u_t_cos * feature  # (..., d)
        f_t = torch.cat([u_t_sin, f_cos], dim=-1)
        return f_t

    def split_t_f(self, f_embedded):
        # feature: (..., d + 1)
        # always regard the first value of the last dim as the embedded time step
        # transform it back to unified time and its feature
        f_t = f_embedded[..., 0]

        # unified time
        t = torch.arcsin(f_t)
        t = t * self.rate / torch.pi
        t = (t + 1.0) / 2.0
        t_logit = torch.logit(t, eps=1e-9)
        # feature without timestep embeddings
        f = torch.div(f_embedded[..., 1:], torch.sqrt(1.0 - f_t ** 2).unsqueeze(-1))
        return t, t_logit, f

    def feature_time(self, f_embedded):
        # feature: (..., d + 1)
        # always regard the first value of the last dim as the embedded time step
        # transform it back to unified time
        f_t = f_embedded[..., 0]
        t = torch.arcsin(f_t)
        t = t * self.rate / torch.pi
        t = (t + 1.0) / 2.0
        return t


class mereNLL(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
        self.one_minus_eps = 1.0 - eps
        self.log_one_minus_eps = log(1.0 - eps)

    def forward(self, x):
        one_minus_eps_mul_x = x * self.one_minus_eps
        coe = self.eps + one_minus_eps_mul_x
        v_log = coe * torch.log(torch.div(coe, one_minus_eps_mul_x))
        return v_log + self.log_one_minus_eps


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], act_fn='relu', active_last=False):
        super().__init__()
        assert act_fn in ['relu', 'tanh', None, '']
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, j in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(i, j))
            if act_fn == 'relu':
                layers.append(nn.ReLU())
            if act_fn == 'tanh':
                layers.append(nn.Tanh())
        if active_last:
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        return self.net(x)


# Resnet Blocks
class CondResnetBlockFC(nn.Module):
    """
    Fully connected Conditional ResNet Block class.
    :param size_h (int): hidden dimension
    :param size_c (int): latent dimension
    """

    def __init__(self, size_h, size_c, beta=0):
        super().__init__()

        # Main Branch
        self.fc_0 = nn.Linear(size_h, size_h)
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")

        self.ln_0 = nn.LayerNorm(size_h)
        self.ln_0.bias.data.zero_()
        self.ln_0.weight.data.fill_(1.0)

        self.fc_1 = nn.Linear(size_h, size_h)
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        self.ln_1 = nn.LayerNorm(size_h)
        self.ln_1.bias.data.zero_()
        self.ln_1.weight.data.fill_(1.0)

        # Conditional Branch
        self.c_fc_0 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.c_fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.c_fc_0.weight, a=0, mode="fan_in")

        self.c_fc_1 = nn.Linear(size_c, size_h)
        nn.init.constant_(self.c_fc_1.bias, 0.0)
        nn.init.kaiming_normal_(self.c_fc_1.weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, x, c, last_activation=True):
        h = self.fc_0(x)
        h = self.ln_0(h * self.c_fc_0(c))
        h = self.activation(h)

        h = self.fc_1(h)
        h = self.ln_1(h * self.c_fc_1(c))

        out = x + h
        if last_activation:
            out = self.activation(out)

        return out
    

class indexRemap:
    def __init__(self, n_index):
        self.n_index = n_index
        self.remap = torch.eye(n=n_index)
        self.sets = [[i] for i in range(n_index)]
    
    def update_remap(self, sets):
        self.sets = sets
        self.remap = torch.zeros_like(self.remap)
        for set_ in sets:
            # set remap_map's prob
            self.remap[torch.tensor(set_).repeat(len(set_)), torch.tensor(set_).repeat_interleave(len(set_))] = 1.0 / len(set_)
            
        return
    
    def sample_remap(self, indices):
        shape = indices.shape
        p_indices = self.remap[indices]
        p_indices = p_indices.view(-1, self.n_index)
        
        # sampling new indices based on probability
        remapped_indices = torch.multinomial(p_indices, 1)
        return remapped_indices.view(shape)


if __name__ == '__main__':
    index_remap = indexRemap(n_index=10)
    print(index_remap.remap)
    index_remap.update_remap([[4, 1, 3], [2, 0, 6, 8], [5, 9], [7]])
    print(index_remap.remap)
    remapped_indices = index_remap.sample_remap(indices=torch.tensor([[3, 5], [2, 9]]))
    print(remapped_indices)
    
    print('TEST clip_segment(input: np.ndarray, t_begin: int, len: int, padding=0.0) -> np.ndarray:')
    test_input1 = np.random.rand(5,2)
    print(test_input1)
    for t in range(10):
        print(f't = {t}')
        seg = clip_segment(test_input1, t, 3, 0.0)
        print(f'seg: {seg}')
    



