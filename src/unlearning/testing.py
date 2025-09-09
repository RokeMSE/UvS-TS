from pa_ewc import PopulationAwareEWC
from src.unlearning.t_gr import *

# Quick test for T-GR
def test_t_gr():
    tgr = TemporalGenerativeReplay("stgcn")
    fake_data = torch.randn(4, 50, 10)  # batch, time, nodes
    surrogate = tgr.perform_temporal_generative_replay(model, fake_data, faulty_indices=3)
    print(f"Original vs Surrogate difference: {torch.norm(fake_data - surrogate)}")

# Quick test for PA-EWC  
def test_pa_ewc():
    pa_ewc = PopulationAwareEWC("stgcn")
    fake_data1 = torch.randn(8, 50, 10)
    fake_data2 = torch.randn(8, 50, 10)
    loss = pa_ewc.calculate_L_pop(fake_data1, fake_data2)
    print(f"L_pop computed: {loss.item()}")