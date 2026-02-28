"""
Step-by-step comparison
"""

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Very simple test
H, MN, L, C = 1, 2, 3, 1
device = 'cuda'

print("="*60)
print("Step-by-step comparison")
print("="*60)

# Simple test data
torch.manual_seed(123)
dtA_real = torch.tensor([[-0.1, -0.2]], device=device)  # (H=1, MN=2)
dtA_imag = torch.tensor([[0.3, 0.4]], device=device)
dtA = torch.complex(dtA_real, dtA_imag)

C_disc_real = torch.tensor([[[0.5, 0.6]]], device=device)  # (C=1, H=1, MN=2)
C_disc_imag = torch.tensor([[[0.7, 0.8]]], device=device)
C_disc = torch.complex(C_disc_real, C_disc_imag)

print(f"\nInputs:")
print(f"  dtA shape: {dtA.shape}")
print(f"  dtA: {dtA}")
print(f"  C_disc shape: {C_disc.shape}")
print(f"  C_disc: {C_disc}")

print("\n" + "="*60)
print("PyTorch implementation:")
print("="*60)

# Step 1: dtA * arange(L)
arange_L = torch.arange(L, device=device, dtype=torch.float32)
print(f"\nStep 1: arange(L) = {arange_L}")

K_all = dtA.unsqueeze(-1) * arange_L  # (H, MN, L)
print(f"  dtA.unsqueeze(-1) shape: {dtA.unsqueeze(-1).shape}")
print(f"  K_all = dtA.unsqueeze(-1) * arange(L)")
print(f"  K_all shape: {K_all.shape}")
print(f"  K_all:\n{K_all}")

# Step 2: exp(K_all)
exp_K_all = torch.exp(K_all)
print(f"\nStep 2: exp_K_all = exp(K_all)")
print(f"  exp_K_all shape: {exp_K_all.shape}")
print(f"  exp_K_all:\n{exp_K_all}")

# Step 3: einsum
print(f"\nStep 3: einsum('chn, hnl -> chl', C_disc, exp_K_all)")
print(f"  C_disc shape: {C_disc.shape} (c={C}, h={H}, n={MN})")
print(f"  exp_K_all shape: {exp_K_all.shape} (h={H}, n={MN}, l={L})")

result = torch.einsum("chn, hnl -> chl", C_disc, exp_K_all)
print(f"  einsum result shape: {result.shape}")
print(f"  einsum result:\n{result}")

K_pytorch = 2 * result.real
print(f"\nStep 4: K = 2 * result.real")
print(f"  K_pytorch:\n{K_pytorch}")

print("\n" + "="*60)
print("Manual calculation (what CUDA should compute):")
print("="*60)

K_manual = torch.zeros(C, H, L, device=device)

for c in range(C):
    for h in range(H):
        for l in range(L):
            sum_val = 0.0 + 0.0j
            for mn in range(MN):
                # Get values
                dtA_val = dtA[h, mn]
                C_val = C_disc[c, h, mn]
                
                # Compute exp(dtA * l)
                exp_val = torch.exp(dtA_val * l)
                
                # Accumulate
                sum_val += C_val * exp_val
                
                print(f"  c={c}, h={h}, l={l}, mn={mn}:")
                print(f"    dtA[{h},{mn}] = {dtA_val}")
                print(f"    C_disc[{c},{h},{mn}] = {C_val}")
                print(f"    exp(dtA*{l}) = {exp_val}")
                print(f"    C * exp = {C_val * exp_val}")
                print(f"    sum so far = {sum_val}")
            
            K_manual[c, h, l] = 2 * sum_val.real
            print(f"  Final K[{c},{h},{l}] = 2 * {sum_val.real} = {K_manual[c, h, l]}")

print(f"\nK_manual:\n{K_manual}")
print(f"\nK_pytorch:\n{K_pytorch}")
print(f"\nDifference: {torch.abs(K_manual - K_pytorch).max().item()}")

print("\n" + "="*60)
print("Now test CUDA:")
print("="*60)

from fused_ssm import fused_ssm_forward
K_cuda = fused_ssm_forward(dtA, C_disc, L)
print(f"K_cuda:\n{K_cuda}")
print(f"\nDifference from PyTorch: {torch.abs(K_cuda - K_pytorch).max().item()}")

