#include "tensor.h"
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <memory>

int main() {
    // Create tensor from nested vector (2D example)
    std::vector<std::vector<float>> data2d = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create tensor from nested vector
    Tensor<float> t1(data2d, handle);
    
    std::cout << "Shape: ";
    for (auto dim : t1.get_shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\nStrides: ";
    for (auto stride : t1.get_strides()) {
        std::cout << stride << " ";
    }
    std::cout << "\nTotal size: " << t1.size() << std::endl;
    
    // Test flat index computation
    std::cout << "Flat index of [1,2]: " << t1.compute_flat_index({1, 2}) << std::endl;

    // Create from flat vector with explicit shape
    std::vector<float> flat = {1, 2, 3, 4, 5, 6};
    Tensor<float> t2(flat, {2, 3}, handle);

    // Matrix multiplication example
    std::vector<std::vector<float>> a = {{1, 2}, {3, 4}};
    std::vector<std::vector<float>> b = {{5, 6}, {7, 8}};
    
    Tensor<float> ta(a, handle);
    Tensor<float> tb(b, handle);
    Tensor<float> tc = ta.matmul(tb);

    auto result = tc.to_host();
    std::cout << "Matrix multiplication result:\n";
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            std::cout << result[i * 2 + j] << " ";
        }
        std::cout << "\n";
    }

    cublasDestroy(handle);
    return 0;
}