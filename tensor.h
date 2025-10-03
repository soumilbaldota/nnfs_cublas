#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>

template<typename T>
class Tensor {
private:
    T* d_data;           // Device pointer
    std::vector<size_t> shape;
    std::vector<size_t> strides;  // Strides for each dimension
    size_t total_size;
    cublasHandle_t handle;

    // Helper to compute strides from shape (row-major order)
    std::vector<size_t> compute_strides(const std::vector<size_t>& dims) {
        std::vector<size_t> str(dims.size());
        if (dims.empty()) return str;
        
        str[dims.size() - 1] = 1;  // Last dimension has stride 1
        for (int i = dims.size() - 2; i >= 0; --i) {
            str[i] = str[i + 1] * dims[i + 1];
        }
        return str;
    }

    // Helper to compute total size from shape
    size_t compute_size(const std::vector<size_t>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1ULL, std::multiplies<size_t>());
    }

    // Get shape from nested vector structure
    template<typename VecType>
    std::vector<size_t> infer_shape(const VecType& vec) {
        std::vector<size_t> dims;
        
        // Handle the case where VecType is already vector<T>
        if constexpr (std::is_same_v<VecType, std::vector<T>>) {
            if (!vec.empty()) {
                dims.push_back(vec.size());
            }
            return dims;
        } else {
            // For nested vectors, recursively get dimensions
            if (!vec.empty()) {
                dims.push_back(vec.size());
                auto inner_dims = infer_shape(vec[0]);
                dims.insert(dims.end(), inner_dims.begin(), inner_dims.end());
            }
            return dims;
        }
    }

public:
    template<typename VecType>
    void flatten_recursive(const VecType& vec, T* buffer, size_t& offset) {
        if constexpr (std::is_same_v<typename VecType::value_type, T>) {
            // Base case: innermost vector, copy all elements
            for (const auto& elem : vec) {
                buffer[offset++] = elem;
            }
        } else {
            // Recursive case: iterate through this dimension
            for (const auto& sub_vec : vec) {
                flatten_recursive(sub_vec, buffer, offset);
            }
        }
    }
    

    // Constructor from nested vectors
    template<typename VecType>
    Tensor(const VecType& data, cublasHandle_t cublas_handle = nullptr) 
        : d_data(nullptr), handle(cublas_handle) {
        
        // Infer shape and compute strides
        shape = infer_shape(data);
        strides = compute_strides(shape);
        total_size = compute_size(shape);
        
        // Allocate device memory
        cudaError_t err = cudaMalloc(&d_data, total_size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        
        // Use pinned memory for faster transfer
        T* h_buffer;
        cudaMallocHost(&h_buffer, total_size * sizeof(T));
        
        // Flatten to buffer using shape (no recursion)
        size_t offset = 0;
        flatten_recursive(data, h_buffer, offset);
        
        // Single bulk copy to device
        err = cudaMemcpy(d_data, h_buffer, total_size * sizeof(T), cudaMemcpyHostToDevice);
        cudaFreeHost(h_buffer);
        
        if (err != cudaSuccess) {
            cudaFree(d_data);
            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        }
        
        // Create handle if not provided
        if (!handle) {
            cublasCreate(&handle);
        }
    }

    // Constructor from flat vector and explicit shape
    Tensor(const std::vector<T>& flat_data, const std::vector<size_t>& dims, cublasHandle_t cublas_handle = nullptr)
        : d_data(nullptr), shape(dims), handle(cublas_handle) {
        
        strides = compute_strides(shape);
        total_size = compute_size(shape);
        
        if (flat_data.size() != total_size) {
            throw std::runtime_error("Data size doesn't match shape");
        }
        
        cudaError_t err = cudaMalloc(&d_data, total_size * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
        
        err = cudaMemcpy(d_data, flat_data.data(), total_size * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_data);
            throw std::runtime_error("cudaMemcpy failed");
        }
        
        if (!handle) {
            cublasCreate(&handle);
        }
    }

    ~Tensor() {
        if (d_data) {
            cudaFree(d_data);
        }
    }

    // Move constructor/assignment
    Tensor(Tensor&& other) noexcept 
        : d_data(other.d_data), shape(std::move(other.shape)), 
          strides(std::move(other.strides)), total_size(other.total_size), 
          handle(other.handle) {
        other.d_data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (d_data) cudaFree(d_data);
            d_data = other.d_data;
            shape = std::move(other.shape);
            strides = std::move(other.strides);
            total_size = other.total_size;
            handle = other.handle;
            other.d_data = nullptr;
        }
        return *this;
    }

    // Disable copy
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Getters
    T* data() { return d_data; }
    const T* data() const { return d_data; }
    const std::vector<size_t>& get_shape() const { return shape; }
    const std::vector<size_t>& get_strides() const { return strides; }
    size_t size() const { return total_size; }
    
    // Convert multi-dimensional index to flat index using strides
    size_t compute_flat_index(const std::vector<size_t>& indices) const {
        size_t flat_idx = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            flat_idx += indices[i] * strides[i];
        }
        return flat_idx;
    }

    // Copy data back to host
    std::vector<T> to_host() const {
        std::vector<T> host_data(total_size);
        cudaMemcpy(host_data.data(), d_data, total_size * sizeof(T), cudaMemcpyDeviceToHost);
        return host_data;
    }

    // Matrix multiplication (for 2D tensors)
    Tensor<T> matmul(const Tensor<T>& other) {
        if (shape.size() != 2 || other.shape.size() != 2) {
            throw std::runtime_error("matmul requires 2D tensors");
        }
        if (shape[1] != other.shape[0]) {
            throw std::runtime_error("Incompatible shapes for matmul");
        }

        size_t m = shape[0];
        size_t k = shape[1];
        size_t n = other.shape[1];

        Tensor<T> result({m, n}, handle);

        T alpha = 1.0, beta = 0.0;

        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, m, k, &alpha,
                       other.d_data, n,
                       d_data, k,
                       &beta, result.d_data, n);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, m, k, &alpha,
                       other.d_data, n,
                       d_data, k,
                       &beta, result.d_data, n);
        }

        return result;
    }

private:
    // Constructor for result tensors
    Tensor(const std::vector<size_t>& dims, cublasHandle_t cublas_handle)
        : d_data(nullptr), shape(dims), handle(cublas_handle) {
        strides = compute_strides(shape);
        total_size = compute_size(shape);
        cudaMalloc(&d_data, total_size * sizeof(T));
    }
};
