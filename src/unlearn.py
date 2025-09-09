import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from src.models.stgcn import STGCN
from src.utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import generate_dataset, get_normalized_adj
from unlearning.motif_def import discover_motifs_proxy
from unlearning.pa_ewc import PopulationAwareEWC, save_fim, load_fim
from unlearning.t_gr import TemporalGenerativeReplay, create_surrogate_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "D:/Model"
checkpoint = torch.load(path + "/model.pt", map_location=device)

# Khởi tạo model với config đã lưu
model = STGCN(**checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# Khởi tạo optimizer và load lại trạng thái
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
