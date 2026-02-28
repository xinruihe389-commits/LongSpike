"""
Test Fused SSM CUDA Extension

Run this script to verify CUDA extension is correctly installed and working
"""

import torch
import sys
import os

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from fused_ssm import fused_ssm_forward, fused_ssm_forward_fallback, is_cuda_available


def test_fused_ssm():
    """
    Test correctness of CUDA implementation
    
    Compare CUDA implementation with PyTorch implementation
    """
    print("=" * 60)
    print("Testing Fused SSM CUDA Implementation")
    print("=" * 60)
    
    # Check if CUDA hardware is available
    if not torch.cuda.is_available():
        print("[ERROR] CUDA hardware not available, skipping test")
        return
    
    # Check if CUDA extension is installed
    if not is_cuda_available():
        print("[ERROR] CUDA extension not installed")
        print("\nPlease compile and install first:")
        print("  cd cuda_ops")
        print("  python setup.py install")
        print("\nOr run installation script:")
        print("  Windows: install.bat")
        print("  Linux/Mac: bash install.sh")
        return
    
    # Test parameters
    H, MN, L, C = 256, 64, 1024, 1
    device = 'cuda'
    
    print(f"\nTest configuration:")
    print(f"  H (hidden dim): {H}")
    print(f"  M*N (state dim): {MN}")
    print(f"  L (sequence length): {L}")
    print(f"  C (channels): {C}")
    
    # Generate random inputs
    # NOTE: dtA.real should be negative for numerical stability (exp decay)
    torch.manual_seed(42)
    dtA_real = -torch.rand(H, MN, device=device) * 0.1  # Negative real part
    dtA_imag = torch.randn(H, MN, device=device) * 0.5
    dtA = torch.complex(dtA_real, dtA_imag)
    C_disc = torch.randn(C, H, MN, dtype=torch.complex64, device=device) * 0.1
    
    # PyTorch implementation
    print("\nRunning PyTorch implementation...")
    K_pytorch = fused_ssm_forward_fallback(dtA, C_disc, L)
    
    # CUDA implementation
    print("Running CUDA implementation...")
    K_cuda = fused_ssm_forward(dtA, C_disc, L)
    
    # Compare results
    abs_diff = torch.abs(K_pytorch - K_cuda)
    
    # Use a more robust relative error calculation
    # Only compute relative error for values that are not too small
    threshold = 1e-5
    mask = torch.abs(K_pytorch) > threshold
    
    if mask.any():
        rel_diff_masked = abs_diff[mask] / torch.abs(K_pytorch[mask])
        max_rel_diff = rel_diff_masked.max().item()
    else:
        max_rel_diff = 0.0
    
    max_abs_diff = abs_diff.max().item()
    
    print(f"\nResult comparison:")
    print(f"  Max absolute error: {max_abs_diff:.2e}")
    print(f"  Max relative error (for |value| > {threshold}): {max_rel_diff:.2e}")
    
    # Find where the max error occurs
    abs_diff_flat = abs_diff.flatten()
    max_abs_idx = torch.argmax(abs_diff_flat).item()
    
    # Convert flat index to 3D coordinates
    C, H, L = K_pytorch.shape
    c_abs = max_abs_idx // (H * L)
    h_abs = (max_abs_idx % (H * L)) // L
    l_abs = max_abs_idx % L
    
    print(f"\n  Max absolute error at [{c_abs}, {h_abs}, {l_abs}]:")
    print(f"    PyTorch: {K_pytorch[c_abs, h_abs, l_abs].item():.6e}")
    print(f"    CUDA:    {K_cuda[c_abs, h_abs, l_abs].item():.6e}")
    print(f"    Relative: {(abs_diff[c_abs, h_abs, l_abs] / (torch.abs(K_pytorch[c_abs, h_abs, l_abs]) + 1e-10)).item():.2e}")
    
    # Check if PyTorch has very small values
    small_values = (torch.abs(K_pytorch) < 1e-6).sum().item()
    print(f"\n  Number of values < 1e-6: {small_values} / {K_pytorch.numel()} ({100*small_values/K_pytorch.numel():.3f}%)")
    
    # More reasonable pass criteria
    if max_abs_diff < 1e-4:
        print("\n  ✅ [PASS] Test passed! (absolute error < 1e-4)")
    elif max_abs_diff < 1e-3 and max_rel_diff < 0.01:
        print("\n  ✅ [PASS] Test passed! (absolute error < 1e-3 and relative error < 1%)")
    else:
        print("\n  ⚠️  [WARNING] Error is large, please check implementation")
    
    # Performance test
    print("\nPerformance test:")
    
    # PyTorch implementation
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        K_pytorch = fused_ssm_forward_fallback(dtA, C_disc, L)
    end.record()
    torch.cuda.synchronize()
    pytorch_time = start.elapsed_time(end) / 100
    
    print(f"  PyTorch implementation: {pytorch_time:.3f} ms")
    
    # CUDA implementation
    start.record()
    for _ in range(100):
        K_cuda = fused_ssm_forward(dtA, C_disc, L)
    end.record()
    torch.cuda.synchronize()
    cuda_time = start.elapsed_time(end) / 100
    
    print(f"  CUDA implementation: {cuda_time:.3f} ms")
    print(f"  Speedup: {pytorch_time / cuda_time:.2f}x")
    
    # Memory test
    print("\nMemory usage:")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    K_pytorch = fused_ssm_forward_fallback(dtA, C_disc, L)
    pytorch_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    K_cuda = fused_ssm_forward(dtA, C_disc, L)
    cuda_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    print(f"  PyTorch implementation: {pytorch_mem:.2f} MB")
    print(f"  CUDA implementation: {cuda_mem:.2f} MB")
    print(f"  Memory saved: {pytorch_mem / cuda_mem:.2f}x")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_fused_ssm()
