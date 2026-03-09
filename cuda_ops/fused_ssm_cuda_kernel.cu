#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>  // for std::min

#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t err = call;                           \
        if (err != cudaSuccess) {                         \
            printf("CUDA error %s:%d: %s\n",              \
                   __FILE__, __LINE__,                    \
                   cudaGetErrorString(err));              \
            exit(1);                                      \
        }                                                 \
    } while (0)

// Helper function for min (works on both host and device)
template<typename T>
__host__ __device__ inline T min_val(T a, T b) {
    return (a < b) ? a : b;
}

#define TILE_MN 64  // Tile size for MN dimension (adjust based on your typical MN size)
#define MAX_THREADS 256


/* ===================== Optimized Forward Kernel ===================== */

__global__ void fused_ssm_forward_kernel(
    const float* __restrict__ dtA_real,
    const float* __restrict__ dtA_imag,
    const float* __restrict__ C_disc_real,
    const float* __restrict__ C_disc_imag,
    float* __restrict__ K,
    int H, int MN, int L, int C
) {
    // Thread organization: 
    // blockIdx.x -> h (feature dimension)
    // blockIdx.y -> c (channel dimension)
    // blockIdx.z -> l_block (sequence length blocks)
    // threadIdx.x -> l (sequence position within block)
    
    int h = blockIdx.x;
    int c = blockIdx.y;
    int l = threadIdx.x + blockIdx.z * blockDim.x;
    
    if (h >= H || c >= C || l >= L) return;
    
    // Shared memory for caching dtA and C_disc
    __shared__ float s_dtA_real[TILE_MN];
    __shared__ float s_dtA_imag[TILE_MN];
    __shared__ float s_C_real[TILE_MN];
    __shared__ float s_C_imag[TILE_MN];
    
    float sum = 0.0f;
    float l_float = (float)l;
    
    // Process MN dimension in tiles
    for (int mn_base = 0; mn_base < MN; mn_base += TILE_MN) {
        int mn_tile = min_val(TILE_MN, MN - mn_base);
        
        // Cooperative loading to shared memory
        // Each thread loads one element (if within bounds)
        if (threadIdx.x < mn_tile) {
            int mn = mn_base + threadIdx.x;
            int dtA_idx = h * MN + mn;
            int C_idx = c * H * MN + h * MN + mn;
            
            s_dtA_real[threadIdx.x] = dtA_real[dtA_idx];
            s_dtA_imag[threadIdx.x] = dtA_imag[dtA_idx];
            s_C_real[threadIdx.x] = C_disc_real[C_idx];
            s_C_imag[threadIdx.x] = C_disc_imag[C_idx];
        }
        __syncthreads();
        
        // Compute using shared memory
        #pragma unroll 8
        for (int i = 0; i < mn_tile; i++) {
            float ar = s_dtA_real[i];
            float ai = s_dtA_imag[i];
            float Cr = s_C_real[i];
            float Ci = s_C_imag[i];
            
            // Compute exp(dtA * l) = exp((ar + i*ai) * l)
            float exp_real_part = ar * l_float;
            float exp_imag_part = ai * l_float;
            
            // exp(ar*l) * (cos(ai*l) + i*sin(ai*l))
            float er = __expf(exp_real_part);
            float sinv, cosv;
            __sincosf(exp_imag_part, &sinv, &cosv);  // Faster than separate sin/cos
            
            float xr = er * cosv;
            float xi = er * sinv;
            
            // Complex multiplication: C_disc * exp(dtA * l)
            sum += Cr * xr - Ci * xi;
        }
        __syncthreads();
    }
    
    // Write result with coalesced memory access
    K[c * H * L + h * L + l] = 2.0f * sum;
}


/* ===================== Optimized Backward Kernel for C_disc ===================== */

__global__ void fused_ssm_backward_C_kernel(
    const float* __restrict__ grad_K,
    const float* __restrict__ dtA_real,
    const float* __restrict__ dtA_imag,
    float* __restrict__ grad_C_disc_real,
    float* __restrict__ grad_C_disc_imag,
    int H, int MN, int L, int C
) {
    // Each thread computes gradient for one (c, h, mn) position
    int mn = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int c = blockIdx.z;
    
    if (mn >= MN || h >= H || c >= C) return;
    
    int d_idx = h * MN + mn;
    float ar = dtA_real[d_idx];
    float ai = dtA_imag[d_idx];
    
    float gr = 0.0f, gi = 0.0f;
    
    // Accumulate over L dimension
    #pragma unroll 4
    for (int l_int = 0; l_int < L; l_int++) {
        float l = (float)l_int;
        int K_idx = c * H * L + h * L + l_int;
        float gk = grad_K[K_idx];
        
        float exp_real_part = ar * l;
        float exp_imag_part = ai * l;
        
        float er = __expf(exp_real_part);
        float sinv, cosv;
        __sincosf(exp_imag_part, &sinv, &cosv);
        
        gr += 2.0f * gk * (er * cosv);
        gi -= 2.0f * gk * (er * sinv);
    }
    
    int out_idx = c * H * MN + h * MN + mn;
    grad_C_disc_real[out_idx] = gr;
    grad_C_disc_imag[out_idx] = gi;
}


/* ===================== Optimized Backward Kernel for dtA ===================== */

__global__ void fused_ssm_backward_dtA_kernel(
    const float* __restrict__ grad_K,
    const float* __restrict__ dtA_real,
    const float* __restrict__ dtA_imag,
    const float* __restrict__ C_disc_real,
    const float* __restrict__ C_disc_imag,
    float* __restrict__ grad_dtA_real,
    float* __restrict__ grad_dtA_imag,
    int H, int MN, int L, int C
) {
    // Each thread computes gradient for one (h, mn) position
    int mn = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    
    if (mn >= MN || h >= H) return;
    
    int idx = h * MN + mn;
    float ar = dtA_real[idx];
    float ai = dtA_imag[idx];
    
    float gr = 0.0f, gi = 0.0f;
    
    // Accumulate over C and L dimensions
    for (int c = 0; c < C; c++) {
        int C_idx = c * H * MN + h * MN + mn;
        float Cr = C_disc_real[C_idx];
        float Ci = C_disc_imag[C_idx];
        
        #pragma unroll 4
        for (int l_int = 0; l_int < L; l_int++) {
            float l = (float)l_int;
            int K_idx = c * H * L + h * L + l_int;
            float gk = grad_K[K_idx];
            
            float exp_real_part = ar * l;
            float exp_imag_part = ai * l;
            
            float er = __expf(exp_real_part);
            float sinv, cosv;
            __sincosf(exp_imag_part, &sinv, &cosv);
            
            // Derivative of exp(dtA*l) w.r.t. dtA
            // d/d(ar): exp(ar*l) * (cos(ai*l) + i*sin(ai*l)) * l
            // d/d(ai): exp(ar*l) * (-sin(ai*l) + i*cos(ai*l)) * l
            
            // For real part (ar):
            float d_exp_real_ar = l * er * cosv;  // Real part of d(exp)/d(ar)
            float d_exp_imag_ar = l * er * sinv;  // Imag part of d(exp)/d(ar)
            
            // For imag part (ai):
            float d_exp_real_ai = -l * er * sinv;  // Real part of d(exp)/d(ai)
            float d_exp_imag_ai = l * er * cosv;   // Imag part of d(exp)/d(ai)
            
            // Complex multiplication: C_disc * d(exp)/d(dtA), then take real part
            gr += 2.0f * gk * (Cr * d_exp_real_ar - Ci * d_exp_imag_ar);
            gi += 2.0f * gk * (Cr * d_exp_real_ai - Ci * d_exp_imag_ai);
        }
    }
    
    grad_dtA_real[idx] = gr;
    grad_dtA_imag[idx] = gi;
}


/* ===================== Host Wrappers ===================== */

void fused_ssm_forward_cuda(
    const float* dtA_real,
    const float* dtA_imag,
    const float* C_disc_real,
    const float* C_disc_imag,
    float* K,
    int H, int MN, int L, int C
) {
    // Optimized thread organization:
    // - Each block handles one (h, c) pair
    // - Threads within block handle different l positions
    int threads_per_block = min_val(L, MAX_THREADS);
    int l_blocks = (L + threads_per_block - 1) / threads_per_block;
    
    dim3 threads(threads_per_block);
    dim3 blocks(H, C, l_blocks);

    fused_ssm_forward_kernel<<<blocks, threads>>>(
        dtA_real, dtA_imag,
        C_disc_real, C_disc_imag,
        K,
        H, MN, L, C
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void fused_ssm_backward_cuda(
    const float* grad_K,
    const float* dtA_real,
    const float* dtA_imag,
    const float* C_disc_real,
    const float* C_disc_imag,
    float* grad_dtA_real,
    float* grad_dtA_imag,
    float* grad_C_disc_real,
    float* grad_C_disc_imag,
    int H, int MN, int L, int C
) {
    // Launch kernel for C_disc gradient
    {
        int threads_per_block = min_val(MN, MAX_THREADS);
        dim3 threads(threads_per_block);
        dim3 blocks((MN + threads_per_block - 1) / threads_per_block, H, C);
        
        fused_ssm_backward_C_kernel<<<blocks, threads>>>(
            grad_K,
            dtA_real, dtA_imag,
            grad_C_disc_real, grad_C_disc_imag,
            H, MN, L, C
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    // Launch kernel for dtA gradient
    {
        int threads_per_block = min_val(MN, MAX_THREADS);
        dim3 threads(threads_per_block);
        dim3 blocks((MN + threads_per_block - 1) / threads_per_block, H);
        
        fused_ssm_backward_dtA_kernel<<<blocks, threads>>>(
            grad_K,
            dtA_real, dtA_imag,
            C_disc_real, C_disc_imag,
            grad_dtA_real, grad_dtA_imag,
            H, MN, L, C
        );
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}
