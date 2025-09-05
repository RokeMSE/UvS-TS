# pip install giotto-tda ???????????????????
import numpy as np
from dtaidistance import dtw

def discover_motifs_proxy(dataset, u, faulty_node_idx, threshold): # Sample parameters...
    """
    The PEPA algorithm, implementing TDA is a research task.
    This function identifies a "concept" as any data involving a USER PROVIDES INFO about what to forget.
    
    Params:
        dataset
        faulty_node_idx (int): The index of the sensor node to be "forgotten". Doesn't need to be an int, or a faulty node, USER PROVIDES INFO.
        window_size (int): Number of samples to treat as the forget set (to limit the scope if need).
                           
    Returns:
        forget_indices (list): Indices of samples related to the f_i.
        retain_indices (list): All other indices.
    """

    # # NOTE: This is a proxy implementation. A real implementation would involve TDA to find motifs.
    # # TASK: TDA implementation, search the whole dataset for a specific pattern.

    # total_samples = len(dataset)
    # all_indices = np.arange(total_samples) # Random ass placeholder logic
    
    # forget_indices = list(range(window_size)) # First `window_size` samples to forget
    # retain_indices = list(range(window_size, total_samples)) # The rest to retain
    
    # return forget_indices, retain_indices
    S = dataset[:, 1, :]
    sensor, time_step = S.shape
    results = []
    idx = 1
    i = 1
    check = False
    pre_value = -1
    while i <= time_step:
        if check == True:
            if S[sensor, i - 1] == pre_value:
                idx += 1
            elif dtw.distance_fast(u, S[sensor, i-idx : i]) <= threshold:
                check = True
                pre_value = S[sensor, i-1]
                idx += 1
            else:
                results.append((sensor, i - idx, idx - 1))
                check = False
                idx = 1
        elif dtw.distance_fast(u, S[sensor, i-idx : i]) <= threshold:
            pre_value = S[sensor, i - 1]
            check = True
            idx += 1

        i += 1

    if check == True:
        results.append((sensor, i - idx, idx - 1))
    
    return results