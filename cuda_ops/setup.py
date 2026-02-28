"""
Setup script for Fused SSM CUDA Extension (Pre-compiled version)

Compile and install:
    python setup.py install

Verify installation:
    python -c "import fused_ssm_cuda; print('Installation successful!')"

Uninstall:
    pip uninstall fused_ssm_cuda -y
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

# Get CUDA architecture
def get_cuda_arch():
    """Auto-detect GPU architecture"""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f'sm_{capability[0]}{capability[1]}'
        print(f"Detected GPU architecture: {arch}")
        return [arch]
    else:
        print("Warning: No CUDA device detected, using default architectures")
        return ['sm_75', 'sm_80', 'sm_86']  # Default supported architectures

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_file = os.path.join(current_dir, 'fused_ssm_kernel.cpp')
cu_file = os.path.join(current_dir, 'fused_ssm_cuda_kernel.cu')

setup(
    name='fused_ssm_cuda',
    version='1.0.0',
    description='Fused SSM CUDA Extension for Spiking SSM',
    author='Your Name',
    ext_modules=[
        CUDAExtension(
            name='fused_ssm_cuda',
            sources=[cpp_file, cu_file],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++14'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--expt-relaxed-constexpr',
                    '-std=c++14',
                ] + [f'-arch={arch}' for arch in get_cuda_arch()]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch>=1.8.0'],
    python_requires='>=3.7',
)



