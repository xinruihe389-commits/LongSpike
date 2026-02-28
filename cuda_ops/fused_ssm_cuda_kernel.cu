#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

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


/* ===================== Forward Kernel ===================== */

__global__ void fused_ssm_forward_kernel(
    const float* __restrict__ dtA_real,
    const float* __restrict__ dtA_imag,
    const float* __restrict__ C_disc_real,
    const float* __restrict__ C_disc_imag,
    float* __restrict__ K,
    int H, int MN, int L, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * L;
    if (idx >= total) return;

    int l_int = idx % L;
    int h = (idx / L) % H;
    int c = idx / (L * H);
    
    float l = (float)l_int;  // Convert to float!

    float sum = 0.0f;

    #pragma unroll 4
    for (int mn = 0; mn < MN; mn++) {
        int dtA_idx = h * MN + mn;
        int C_idx   = c * H * MN + h * MN + mn;

        float ar = dtA_real[dtA_idx];
        float ai = dtA_imag[dtA_idx];
        float Cr = C_disc_real[C_idx];
        float Ci = C_disc_imag[C_idx];

        float er = __expf(ar * l);
        float cosv = __cosf(ai * l);
        float sinv = __sinf(ai * l);

        float xr = er * cosv;
        float xi = er * sinv;

        sum += Cr * xr - Ci * xi;
    }

    K[idx] = 2.0f * sum;
}


/* ===================== Backward Kernel ===================== */

__global__ void fused_ssm_backward_kernel(
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < C * H * MN) {
        int mn = idx % MN;
        int h  = (idx / MN) % H;
        int c  = idx / (MN * H);

        float gr = 0.0f, gi = 0.0f;

        for (int l_int = 0; l_int < L; l_int++) {
            float l = (float)l_int;  // Convert to float!
            int K_idx = c * H * L + h * L + l_int;
            float gk = grad_K[K_idx];

            int d_idx = h * MN + mn;
            float ar = dtA_real[d_idx];
            float ai = dtA_imag[d_idx];

            float er = __expf(ar * l);
            float cosv = __cosf(ai * l);
            float sinv = __sinf(ai * l);

            gr += 2.0f * gk * (er * cosv);
            gi -= 2.0f * gk * (er * sinv);
        }

        grad_C_disc_real[idx] = gr;
        grad_C_disc_imag[idx] = gi;
    }

    if (idx < H * MN) {
        int h  = idx / MN;
        int mn = idx % MN;

        float gr = 0.0f, gi = 0.0f;

        for (int c = 0; c < C; c++) {
            int C_idx = c * H * MN + h * MN + mn;
            float Cr = C_disc_real[C_idx];
            float Ci = C_disc_imag[C_idx];

            for (int l_int = 0; l_int < L; l_int++) {
                float l = (float)l_int;  // Convert to float!
                int K_idx = c * H * L + h * L + l_int;
                float gk = grad_K[K_idx];

                float ar = dtA_real[idx];
                float ai = dtA_imag[idx];

                float er = __expf(ar * l);
                float cosv = __cosf(ai * l);
                float sinv = __sinf(ai * l);

                float dr = l * er * cosv;
                float di = l * er * sinv;

                gr += 2.0f * gk * (Cr * dr - Ci * di);
                gi += 2.0f * gk * (Cr * di + Ci * dr);
            }
        }

        grad_dtA_real[idx] = gr;
        grad_dtA_imag[idx] = gi;
    }
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
    int total = C * H * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

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
    int total = max(C * H * MN, H * MN);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_ssm_backward_kernel<<<blocks, threads>>>(
        grad_K,
        dtA_real, dtA_imag,
        C_disc_real, C_disc_imag,
        grad_dtA_real, grad_dtA_imag,
        grad_C_disc_real, grad_C_disc_imag,
        H, MN, L, C
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
