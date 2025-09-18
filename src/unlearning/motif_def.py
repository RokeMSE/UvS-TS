# pip install tslearn
import numpy as np
import torch
from tslearn.metrics import dtw

def discover_motifs_proxy(dataset, u, faulty_node_idx, threshold):
    """
    This function identifies a "concept" as any data involving a USER PROVIDES INFO about what to forget.
    
    Params:
        dataset: (num_nodes, num_features, num_timesteps).
        u: unlearn data.
        faulty_node_idx (int): The index of the sensor node to be "forgotten". Doesn't need to be an int, or a faulty node, USER PROVIDES INFO.
        threshold (float): DTW distance threshold.
                           
    Returns:
        forget_indices (tuple): (sensor, start, end) of subset similar with u
    """
    S = dataset[:, 1, :]
    _, time_step = S.shape
    forget_indices = []
    idx = 1
    i = 1
    check = False
    pre_value = -1
    
    while i <= time_step:
        if check == True:
            if S[faulty_node_idx, i - 1] == pre_value:
                idx += 1
            elif dtw(u, S[faulty_node_idx, i-idx : i]) <= threshold:
                check = True
                pre_value = S[faulty_node_idx, i-1]
                idx += 1
            else:
                forget_indices.append([i - idx, i - 1])
                check = False
                idx = 1
        elif dtw(u, S[faulty_node_idx, i-idx : i]) <= threshold:
            pre_value = S[faulty_node_idx, i - 1]
            check = True
            idx += 1

        i += 1

    if check == True:
        forget_indices.append([i - idx, i - 1])
    
    retain_indices = []
    if forget_indices[0][0] != 0:
        retain_indices.append([0, forget_indices[0][0]])

    for i in range(1, len(forget_indices)):
        retain_indices.append([forget_indices[i - 1][1], forget_indices[i][0]])

    if forget_indices[-1][1] != time_step:
        retain_indices.append([forget_indices[-1][1], time_step])
    
    return forget_indices, retain_indices