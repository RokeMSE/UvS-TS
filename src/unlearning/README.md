## Integration for `t_gr.py`
- Implement the 'Reconstruction and Neutralization' strategy (needs access to the model being unlearned).
- For a given sample $x_f$ from $D_f$, the function should:
    1. Treat the motif as missing data.
    2. Use the model itself to impute the missing section to get $x_f'$. (STCGN: refer to CSDI)
    3. Generate the final surrogate data. 
        + Using the neutral reconstruction $x_f'$ +
        + Error-minimizing noise (add later).

## Integration for `motif_def.py`
- PEPA Implementation:  
    + Identifying patterns via simple Euclidean distance → ?????
    + Need: 
        1. Persistent homology 
        2. topological data analysis (TDA)
***NOTICE: Use notebooks to visualize the motifs discovered***