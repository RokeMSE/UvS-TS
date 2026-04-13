# pip install stumpy
import numpy as np
import stumpy


def discover_motifs_proxy(dataset, u, faulty_node_idx, threshold):
    """
    Identify temporal windows in the faulty node's series that are
    similar in shape to the user-supplied forget example *u*.

    Uses stumpy.mass (Matrix Profile distance profile) instead of DTW:
      - O(n log n) instead of O(n * m^2)
      - z-normalised Euclidean distance preserves shape similarity
      - Mueen (2014) showed DTW and z-norm Euclidean rankings are
        "practically indistinguishable" for motif discovery

    Threshold is in z-normalised Euclidean space:
      0            = perfect shape match
      ~2*sqrt(m)   = maximally dissimilar (anti-correlated)
    Typical starting values: 1.0 – 3.0  (was 0.05 – 0.5 for DTW).

    Params:
        dataset:         (num_nodes, num_features, num_timesteps)
        u:               1-D array — the forget example (single feature)
        faulty_node_idx: sensor node index
        threshold:       z-norm Euclidean distance threshold

    Returns:
        forget_indices:  list of [start, end] pairs (merged, non-overlapping)
        retain_indices:  complementary list of [start, end] pairs
    """
    S = dataset[:, 0, :]            # (num_nodes, num_timesteps) — use feature 0
    _, time_step = S.shape
    window_size = len(u)

    series = S[faulty_node_idx].astype(np.float64)
    query  = u.astype(np.float64)

    # distance_profile[i] = z-norm Euclidean distance between query and
    # series[i : i + window_size]
    distance_profile = stumpy.mass(query, series)

    forget_indices = []
    for i, dist in enumerate(distance_profile):
        if dist <= threshold:
            # Merge overlapping / adjacent windows
            if forget_indices and i <= forget_indices[-1][1]:
                forget_indices[-1][1] = i + window_size
            else:
                forget_indices.append([i, i + window_size])

    if not forget_indices:
        print("Warning: No motifs found with the given threshold. "
              "Entire dataset is considered 'retain'.")
        return forget_indices, [[0, time_step]]

    retain_indices = []
    last_end = 0
    for start, end in forget_indices:
        if start > last_end:
            retain_indices.append([last_end, start])
        last_end = end
    if last_end < time_step:
        retain_indices.append([last_end, time_step])

    return forget_indices, retain_indices
