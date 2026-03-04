import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

class SpatialPyramid:
    def __init__(self, locations: np.ndarray, height: int = 5):
        """
        Args:
            locations: Array of (latitude, longitude) pairs, shape (num_locations, 2)
            height: Number of levels in the pyramid (H in paper)
        """
        self.height = height
        self.locations = locations
        self.num_locations = len(locations)
        self.grid_assignments = self._build_pyramid()
    
    def _build_pyramid(self) -> List[np.ndarray]:
        """
        Build hierarchical grid assignments for each location.
        Returns list where grid_assignments[h] contains grid indices for level h.
        """
        # Get bounding box
        lat_min, lat_max = self.locations[:, 0].min(), self.locations[:, 0].max()
        lon_min, lon_max = self.locations[:, 1].min(), self.locations[:, 1].max()
        
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        
        # Add small padding to avoid edge cases
        lat_min -= lat_range * 0.01
        lat_max += lat_range * 0.01
        lon_min -= lon_range * 0.01
        lon_max += lon_range * 0.01
        
        grid_assignments = []
        
        for h in range(self.height):
            # Number of grid cells at this level: 2^h x 2^h
            num_cells_per_dim = 2 ** h
            
            # Grid cell size
            lat_cell_size = (lat_max - lat_min) / num_cells_per_dim
            lon_cell_size = (lon_max - lon_min) / num_cells_per_dim
            
            # Assign each location to a grid cell
            assignments = np.zeros(self.num_locations, dtype=np.int64)
            
            for i, (lat, lon) in enumerate(self.locations):
                # Find grid indices
                lat_idx = int((lat - lat_min) / lat_cell_size)
                lon_idx = int((lon - lon_min) / lon_cell_size)
                
                # Clip to valid range
                lat_idx = min(lat_idx, num_cells_per_dim - 1)
                lon_idx = min(lon_idx, num_cells_per_dim - 1)
                
                # Flatten to single index
                cell_idx = lat_idx * num_cells_per_dim + lon_idx
                assignments[i] = cell_idx
            
            grid_assignments.append(assignments)
        
        return grid_assignments
    
    def get_path(self, location_idx: int) -> List[int]:
        """
        Get the hierarchical path from root to leaf for a location.
        Returns list of grid indices [l_1, l_2, ..., l_H].
        """
        return [self.grid_assignments[h][location_idx] for h in range(self.height)]
    
    def get_num_grids(self, level: int) -> int:
        """Get number of grid cells at a given level."""
        return 4 ** level  # 2^level x 2^level grid


class SpatialPyramidParameters(nn.Module):
    def __init__(self, spatial_pyramid: SpatialPyramid, num_topics: int, 
                 num_time_slices: int, param_type: str = "native"):
        """
        Args:
            spatial_pyramid: SpatialPyramid instance
            num_topics: Number of latent topics (K)
            num_time_slices: Number of time slices (T)
            param_type: "native" or "tourist"
        """
        super().__init__()
        
        self.pyramid = spatial_pyramid
        self.num_topics = num_topics
        self.num_time_slices = num_time_slices
        self.param_type = param_type
        
        # Create parameters for each level of pyramid
        # θ^native_{l_h,t} or θ^tourist_{l_h,t} for each level h
        self.level_params = nn.ParameterList()
        
        for h in range(spatial_pyramid.height):
            num_grids = spatial_pyramid.get_num_grids(h)
            # Shape: (num_grids, num_time_slices, num_topics)
            param = nn.Parameter(torch.randn(num_grids, num_time_slices, num_topics) * 0.01)
            self.level_params.append(param)
    
    def get_aggregated_params(self, location_idx: int, time_slice: int) -> torch.Tensor:
        """
        Get aggregated parameters for a location and time using Equation 11.
        θ^native_{l,t} = Σ_{h=1}^H θ^native_{l_h,t}
        
        Args:
            location_idx: Index of the location
            time_slice: Time slice index
        
        Returns:
            Aggregated topic vector of shape (num_topics)
        """
        path = self.pyramid.get_path(location_idx)
        
        # Sum parameters across all levels in the path
        aggregated = torch.zeros(self.num_topics, device=self.level_params[0].device)
        
        for h, grid_idx in enumerate(path):
            aggregated += self.level_params[h][grid_idx, time_slice, :]
        
        return aggregated
    
    def get_batch_aggregated_params(self, location_indices: torch.Tensor, 
                                     time_slices: torch.Tensor) -> torch.Tensor:
        """
        Batch version of get_aggregated_params.
        """
        batch_size = location_indices.shape[0]
        aggregated = torch.zeros(batch_size, self.num_topics, 
                                device=self.level_params[0].device)
        
        for b in range(batch_size):
            loc_idx = location_indices[b].item()
            time_idx = time_slices[b].item()
            aggregated[b] = self.get_aggregated_params(loc_idx, time_idx)
        
        return aggregated