import torch
def replace_target(forget_target, surrogate, Df_indices, forget_indices, faulty_node_idx, num_timesteps_input):

    new_forget_target = forget_target.clone()

    for w in range(new_forget_target.shape[0]):
        global_start = Df_indices[w].item()
        global_target_start = global_start + 12

        for seg_idx, (f_start, f_end) in enumerate(forget_indices):
            surrogate_seg = surrogate[seg_idx] 
            
            overlap_start = max(global_target_start, f_start)
            overlap_end   = min(global_target_start + 4, f_end)
            
            if overlap_start < overlap_end:
                local_start = overlap_start - global_target_start
                local_end   = overlap_end   - global_target_start
                
                surrogate_offset = overlap_start - max(num_timesteps_input, f_start)
                
                values_to_paste = surrogate_seg[surrogate_offset : surrogate_offset + (local_end - local_start)]

                new_forget_target[w, faulty_node_idx, local_start:local_end, :] = values_to_paste

    
    return new_forget_target


def replace_dataset(dataset, surrogate, forget_indices, faultynode_idx, num_timestep_input, forget_output_mask):
    new_dataset = dataset.copy()
    N, F, T = new_dataset.shape
    for seg_idx, (f_start, f_end) in enumerate(forget_indices):
        surrogate_seg = surrogate[seg_idx].permute((1, 0))
        T_seg = surrogate_seg.shape[1]

        if f_end <= num_timestep_input:
            continue
        
        start = max(f_start, num_timestep_input)

        paste_len = min(T_seg, T - start)
        if paste_len <= 0:
            continue

        new_dataset[faultynode_idx, :, start:start+paste_len] = surrogate_seg[:, :paste_len]

    return new_dataset
