/*
 * Fused SSM Kernel - PyTorch C++ Extension
 */

#include <torch/extension.h>
#include <vector>

/* 
 * CUDA function declarations
 * Implemented in fused_ssm_cuda_kernel.cu
 */
void fused_ssm_forward_cuda(
    const float* dtA_real,
    const float* dtA_imag,
    const float* C_disc_real,
    const float* C_disc_imag,
    float* K,
    int H, int MN, int L, int C
);

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
);


/*
 * Forward function - PyTorch interface
 */
torch::Tensor fused_ssm_forward(
    torch::Tensor dtA,
    torch::Tensor C_disc,
    int L
) {
    TORCH_CHECK(dtA.is_cuda(), "dtA must be a CUDA tensor");
    TORCH_CHECK(C_disc.is_cuda(), "C_disc must be a CUDA tensor");
    TORCH_CHECK(dtA.is_contiguous(), "dtA must be contiguous");
    TORCH_CHECK(C_disc.is_contiguous(), "C_disc must be contiguous");
    TORCH_CHECK(dtA.dtype() == torch::kComplexFloat, "dtA must be complex64");
    TORCH_CHECK(C_disc.dtype() == torch::kComplexFloat, "C_disc must be complex64");

    const int H  = dtA.size(0);
    const int MN = dtA.size(1);
    const int C  = C_disc.size(0);

    TORCH_CHECK(C_disc.size(1) == H,  "C_disc dimension mismatch");
    TORCH_CHECK(C_disc.size(2) == MN, "C_disc dimension mismatch");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(dtA.device());

    torch::Tensor K = torch::empty({C, H, L}, options);

    torch::Tensor dtA_real = torch::real(dtA).contiguous();
    torch::Tensor dtA_imag = torch::imag(dtA).contiguous();
    torch::Tensor C_disc_real = torch::real(C_disc).contiguous();
    torch::Tensor C_disc_imag = torch::imag(C_disc).contiguous();

    fused_ssm_forward_cuda(
        dtA_real.data_ptr<float>(),
        dtA_imag.data_ptr<float>(),
        C_disc_real.data_ptr<float>(),
        C_disc_imag.data_ptr<float>(),
        K.data_ptr<float>(),
        H, MN, L, C
    );

    return K;
}


/*
 * Backward function - PyTorch interface
 */
std::vector<torch::Tensor> fused_ssm_backward(
    torch::Tensor grad_K,
    torch::Tensor dtA,
    torch::Tensor C_disc,
    int L
) {
    TORCH_CHECK(grad_K.is_cuda(), "grad_K must be a CUDA tensor");
    TORCH_CHECK(dtA.is_cuda(), "dtA must be a CUDA tensor");
    TORCH_CHECK(C_disc.is_cuda(), "C_disc must be a CUDA tensor");
    TORCH_CHECK(grad_K.is_contiguous(), "grad_K must be contiguous");
    TORCH_CHECK(dtA.is_contiguous(), "dtA must be contiguous");
    TORCH_CHECK(C_disc.is_contiguous(), "C_disc must be contiguous");

    const int H  = dtA.size(0);
    const int MN = dtA.size(1);
    const int C  = C_disc.size(0);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(dtA.device());

    torch::Tensor grad_dtA_real = torch::zeros({H, MN}, options);
    torch::Tensor grad_dtA_imag = torch::zeros({H, MN}, options);
    torch::Tensor grad_C_disc_real = torch::zeros({C, H, MN}, options);
    torch::Tensor grad_C_disc_imag = torch::zeros({C, H, MN}, options);

    torch::Tensor dtA_real = torch::real(dtA).contiguous();
    torch::Tensor dtA_imag = torch::imag(dtA).contiguous();
    torch::Tensor C_disc_real = torch::real(C_disc).contiguous();
    torch::Tensor C_disc_imag = torch::imag(C_disc).contiguous();

    fused_ssm_backward_cuda(
        grad_K.data_ptr<float>(),
        dtA_real.data_ptr<float>(),
        dtA_imag.data_ptr<float>(),
        C_disc_real.data_ptr<float>(),
        C_disc_imag.data_ptr<float>(),
        grad_dtA_real.data_ptr<float>(),
        grad_dtA_imag.data_ptr<float>(),
        grad_C_disc_real.data_ptr<float>(),
        grad_C_disc_imag.data_ptr<float>(),
        H, MN, L, C
    );

    return {
        torch::complex(grad_dtA_real, grad_dtA_imag),
        torch::complex(grad_C_disc_real, grad_C_disc_imag)
    };
}


/*
 * PyBind11 module binding
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_ssm_forward, "Fused SSM forward (CUDA)");
    m.def("backward", &fused_ssm_backward, "Fused SSM backward (CUDA)");
}
