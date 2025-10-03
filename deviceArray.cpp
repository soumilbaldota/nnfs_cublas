#include <cuda_runtime.h>
#include <stdexcept>

template <typename T>
class DeviceArray {
public:
    T* data;
    size_t size;
    DeviceArray(size_t n) : size(n) {
        cudaError_t err = cudaMalloc(&data, n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    ~DeviceArray() {
        cudaFree(data);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
};
