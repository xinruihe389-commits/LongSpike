"""
Fused SSM CUDA Extension - Python Interface (Pre-compiled version)

This module provides Python interface for fused SSM kernel, including:
1. Import pre-compiled CUDA module
2. PyTorch Autograd Function wrapper
3. Convenient call interface

Before use, please compile:
    cd cuda_ops
    python setup.py install
"""

import torch
import torch.nn as nn
import warnings

# Global variable: CUDA extension module
_fused_ssm_cuda = None


def _load_cuda_extension():
    """
    Load pre-compiled CUDA extension
    """
    global _fused_ssm_cuda
    
    if _fused_ssm_cuda is not None:
        return _fused_ssm_cuda
    
    try:
        # Import pre-compiled module
        import fused_ssm_cuda
        _fused_ssm_cuda = fused_ssm_cuda
        print("CUDA extension loaded successfully!")
        return _fused_ssm_cuda
        
    except ImportError as e:
        warnings.warn(
            f"CUDA extension not installed: {e}\n"
            f"Please compile and install first:\n"
            f"  cd cuda_ops\n"
            f"  python setup.py install\n"
            f"Will use PyTorch fallback implementation"
        )
        return None


def is_cuda_available():
    """Check if CUDA extension is available"""
    global _fused_ssm_cuda
    if _fused_ssm_cuda is None:
        _fused_ssm_cuda = _load_cuda_extension()
    return _fused_ssm_cuda is not None


class FusedSSMFunction(torch.autograd.Function):
    """
    Fused SSM Autograd Function
    
    This class implements custom forward and backward,
    using CUDA kernel for efficient computation.
    """
    
    @staticmethod
    def forward(ctx, dtA, C_disc, L):
        """
        Forward pass
        
        Args:
            dtA: Complex tensor (H, M*N_state)
            C_disc: Complex tensor (C, H, M*N_state)
            L: Sequence length (int)
        
        Returns:
            K: Real tensor (C, H, L)
        """
        # Check CUDA extension
        cuda_ext = _load_cuda_extension()
        if cuda_ext is None:
            raise RuntimeError("CUDA extension not available")
        
        # Ensure inputs are contiguous
        dtA = dtA.contiguous()
        C_disc = C_disc.contiguous()
        
        # Call CUDA kernel
        K = cuda_ext.forward(dtA, C_disc, L)
        
        # Save for backward
        ctx.save_for_backward(dtA, C_disc)
        ctx.L = L
        
        return K
    
    @staticmethod
    def backward(ctx, grad_K):
        """
        Backward pass
        
        Args:
            grad_K: Output gradient (C, H, L)
        
        Returns:
            grad_dtA: Gradient of dtA (H, M*N_state)
            grad_C_disc: Gradient of C_disc (C, H, M*N_state)
            None: L has no gradient
        """
        # Check CUDA extension
        cuda_ext = _load_cuda_extension()
        if cuda_ext is None:
            raise RuntimeError("CUDA extension not available")
        
        # Get saved tensors
        dtA, C_disc = ctx.saved_tensors
        L = ctx.L
        
        # Ensure gradient is contiguous
        grad_K = grad_K.contiguous()
        
        # Call CUDA kernel
        grad_dtA, grad_C_disc = cuda_ext.backward(grad_K, dtA, C_disc, L)
        
        return grad_dtA, grad_C_disc, None


def fused_ssm_forward(dtA, C_disc, L):
    """
    Convenient forward function
    
    Args:
        dtA: Complex tensor (H, M*N_state)
        C_disc: Complex tensor (C, H, M*N_state)
        L: Sequence length
    
    Returns:
        K: Real tensor (C, H, L)
    """
    return FusedSSMFunction.apply(dtA, C_disc, L)


def fused_ssm_forward_fallback(dtA, C_disc, L):
    """
    PyTorch fallback implementation (when CUDA is not available)
    """
    device = dtA.device
    dtype = dtA.real.dtype
    
    # Original implementation
    K_all = dtA.unsqueeze(-1) * torch.arange(L, device=device, dtype=dtype)
    exp_K_all = torch.exp(K_all)
    K = 2 * torch.einsum("chn, hnl -> chl", C_disc, exp_K_all).real
    
    return K



