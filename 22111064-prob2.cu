// nvcc -g -G -arch=sm_61 -std=c++11 22111064-prob2.cu -o 22111064-prob2

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);

using std::cerr;
using std::cout;
using std::endl;

#define BLOCKDIM 16


__global__ void kernel1(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0;
    if (i < N && j < N)
    {
        for (int k = 0; k < N; k++)
        {
            sum += d_A[i * N + k] * d_B[k * N + j];
        }
    }
    d_C[i * N + j] = sum;
}

__global__ void kernel2(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
      // Allocate shared Memory
    __shared__ uint64_t A[BLOCKDIM * BLOCKDIM];
    __shared__ uint64_t B[BLOCKDIM * BLOCKDIM];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    int tileDim = blockDim.x;

    float sum = 0.0;

    if (row < N && col < N)
    {
        for (int i = 0;i < ((N + tileDim - 1) / tileDim); i++)
        {
            A[t_y * tileDim + t_x] = d_A[(row * N) + (i * tileDim) + t_x];
            B[t_y * tileDim + t_x] = d_B[(i * tileDim + t_y) * N + col];

            // Every single threads must come to this point before continue
            __syncthreads();

            // Operation on tile
            for (int j = 0;j < tileDim; j++)
            {
                if (j + (i * tileDim) < N)
                {
                    sum += A[t_y * tileDim + j] * B[j * tileDim + t_x];
                }
            }
            __syncthreads();
        }
        d_C[row * N + col] = sum;
    }
}

__host__ void cpumatMul(const uint64_t* h_A, const uint64_t* h_B, uint64_t* h_C) {
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            float sum = 0.0;
            for (uint64_t k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C[i * N + j] = sum;
        }
    }
}

__host__ void check_result(const uint64_t* w_ref, const uint64_t* w_opt) {
    bool wrong = false;
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            if (w_ref[i * N + j] != w_opt[i * N + j]) {
                wrong = true;
                goto out;
            }
        }
    }
out:
    if (wrong) {
        cout << " Diffs found!" << endl;
    }
    else {
        cout << "No differences found between base and test versions\n";
    }
}

double rtclock() { // Seconds
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int main() {
    const uint64_t SIZE = N * N;

    uint64_t* h_A, * h_B, * h_cpu_C, * h_gpu1_C, * h_gpu2_C;

    h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            h_A[i * N + j] = rand() % 64;
            h_B[i * N + j] = 2;
            h_cpu_C[i * N + j] = 0;
            h_gpu1_C[i * N + j] = 0;
            h_gpu2_C[i * N + j] = 0;
        }
    }

    double clkbegin = rtclock();
    cpumatMul(h_A, h_B, h_cpu_C);
    double clkend = rtclock();
    double cpu_time = clkend - clkbegin;
    cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

    cudaError_t status;
    cudaEvent_t start, end;

    uint64_t* d_A, * d_B, * d_C1;
    status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
    if (status != cudaSuccess) {
        cerr << cudaGetErrorString(status) << endl;
    }
    status = cudaMalloc(&d_B, SIZE * sizeof(uint64_t));
    status = cudaMalloc(&d_C1, SIZE * sizeof(uint64_t));

    status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
    status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // TODO: Fill in
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 block(BLOCKDIM, BLOCKDIM);
    int blocksInGridRow = ceil(N / (float)BLOCKDIM);
    dim3 grid(blocksInGridRow, blocksInGridRow);

    // Marking start time
    cudaEventRecord(start);

    // launching Kernal1
    kernel1 << < grid, block >> > (d_A, d_B, d_C1);
    cudaDeviceSynchronize();

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);


    cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    check_result(h_cpu_C, h_gpu1_C);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

    uint64_t* d_C2;
    status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_t));

    // TODO: Fill in
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    //  cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

      // Marking start time
    cudaEventRecord(start);

    // launching Kernal1
    kernel2 << < grid, block >> > (d_A, d_B, d_C2);
    cudaDeviceSynchronize();

    // Marking end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    check_result(h_cpu_C, h_gpu2_C);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);

    free(h_A);
    free(h_B);
    free(h_cpu_C);
    free(h_gpu1_C);
    free(h_gpu2_C);

    return EXIT_SUCCESS;
}
