#include "deviceArray.cpp"
#include <cublas_v2.h>
#include <iostream>

int main() {
    vector<vector<double>> weights{{1,2,3}, {1,2.5,3.5}, {4,1,2.5}};
    vector<double> input{1,2,3};
    double bias = {3,2,1};
    int N = input.size();

    weights = DeviceArray()
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int _w = 0; _w < weights.size(); _w++) {
        double res = bias[_w];

        DeviceArray<float> d_weights(N);
        DeviceArray<float> d_input(N);

        cudaMemcpy(d_weights.data, weights[_w].data(),
                N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input.data, input.data(),
                N * sizeof(float), cudaMemcpyHostToDevice);

        float result = 0.0f;

        cublasSdot(handle, N,
                d_weights.data, 1,
                d_input.data, 1,
                &result);

        res += result;
    }

    cublasDestroy(handle);


    return res;
}