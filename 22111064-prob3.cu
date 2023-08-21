// nvcc -g -G -arch=sm_61 -std=c++11 22111064-prob3.cu -o 22111064-prob3

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);

using std::cerr;
using std::cout;
using std::endl;
#define BLOCKDIM 16

__global__ void opt_matmul_prob2(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
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

__global__ void pinned(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
        // TODO: Fill in
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

__global__ void zerocopy(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
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

__global__ void uvm(const uint64_t* d_A, const uint64_t* d_B, uint64_t* d_C) {
    // TODO: Fill in
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
__host__ double rtclock() { // Seconds
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

    uint64_t* h_A, * h_B, * h_cpu_C, * h_gpu1_C;

    h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
    h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            h_A[i * N + j] = rand() % 64;
            h_B[i * N + j] = 2;
            h_cpu_C[i * N + j] = 0;
            h_gpu1_C[i * N + j] = 0;
        }
    }

    /*-------------------------------------------Serial--------------------------------------------*/
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



    // TODO: Fill in
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 block(BLOCKDIM, BLOCKDIM);
    int blocksInGridRow = ceil(N / (float)BLOCKDIM);
    dim3 grid(blocksInGridRow, blocksInGridRow);

    /*-------------------------------------------Prob 2--------------------------------------------*/
    // Marking start time
    cudaEventRecord(start);

    status = cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
    status = cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // launching kernal opt_matmul_prob2
    opt_matmul_prob2 <<< grid, block >>> (d_A, d_B, d_C1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Check results
    check_result(h_cpu_C, h_gpu1_C);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

    //DeAllocating unessential memory
    cudaFree(d_C1);
    free(h_B);
    free(h_gpu1_C);

    /*-------------------------------------------Pinned--------------------------------------------*/

    uint64_t* d_C2, * h_gpu2_C, * h_A2;
    status = cudaMalloc(&d_C2, SIZE * sizeof(uint64_t));

    // TODO: Fill in

    // Allocating pinned memory
    status = cudaHostAlloc(&h_A2, SIZE * sizeof(uint64_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_B, SIZE * sizeof(uint64_t), cudaHostAllocDefault);
    cudaHostAlloc(&h_gpu2_C, SIZE * sizeof(uint64_t), cudaHostAllocDefault);

    // Initialising pinned memory
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {

            h_A2[i * N + j] = h_A[i * N + j];
            h_B[i * N + j] = 2;
            h_gpu2_C[i * N + j] = 0;
        }
    }
    
    // Marking start time
    cudaEventRecord(start);

    // Copy data from the CPU to GPU
    cudaMemcpy(d_A, h_A2, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice);


    // launching Kernal pinned
    pinned <<< grid, block >>> (d_A, d_B, d_C2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Marking end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Check results
    check_result(h_cpu_C, h_gpu2_C);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

    //DeAllocating unessential memory
    cudaFreeHost(h_A2);
    cudaFreeHost(h_B);
    cudaFreeHost(h_gpu2_C);
    cudaFree(d_C2);

    /*-------------------------------------------Zero Copy-----------------------------------------*/

    uint64_t* h_gpu3_C, * h_A3, * d_A3, * d_B3, * d_C3;
    cudaHostAlloc(&h_A3, SIZE * sizeof(uint64_t), cudaHostAllocMapped);
    cudaHostAlloc(&h_B, SIZE * sizeof(uint64_t), cudaHostAllocMapped);
    cudaHostAlloc(&h_gpu3_C, SIZE * sizeof(uint64_t), cudaHostAllocMapped);
    // Initialising memory
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            h_B[i * N + j] = 2;
            h_A3[i * N + j] = h_A[i * N + j];
        }
    }

    // pass the pointer to device
    cudaHostGetDevicePointer(&d_A3, h_A3, 0);
    cudaHostGetDevicePointer(&d_B3, h_B, 0);
    cudaHostGetDevicePointer(&d_C3, h_gpu3_C, 0);

    // Marking start time
    cudaEventRecord(start);

    zerocopy << < grid, block >> > (d_A3, d_B3, d_C3);
    cudaDeviceSynchronize();

    // Marking end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Check results
    check_result(h_cpu_C, h_gpu3_C);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 3 time (ms): " << kernel_time << "\n";

    cudaFreeHost(h_A3);
    cudaFreeHost(h_B);
    cudaFreeHost(h_gpu3_C);
    cudaFree(d_A3);
    cudaFree(d_B3);
    cudaFree(d_C3);


    /*----------------------------------Unified virtual memory-------------------------------------*/

    uint64_t* h_gpu4_C, * h_A4;

    // Allocation memory for these pointers
    cudaMallocManaged(&h_A4, SIZE * sizeof(uint64_t));
    cudaMallocManaged(&h_B, SIZE * sizeof(uint64_t));
    cudaMallocManaged(&h_gpu4_C, SIZE * sizeof(uint64_t));

    // Initialising Unified virtual Memory
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j < N; j++) {
            h_A4[i * N + j] = h_A[i * N + j];
            h_B[i * N + j] = 2;
            h_gpu4_C[i * N + j] = 0;
        }
    }

    // Marking start time
    cudaEventRecord(start);

    uvm << < grid, block >> > (h_A4, h_B, h_gpu4_C);
    cudaDeviceSynchronize();

    // Marking end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Check results
    check_result(h_cpu_C, h_gpu4_C);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 4 time (ms): " << kernel_time << "\n";

    cudaFree(h_A4);
    cudaFree(h_gpu4_C);

    /*----------------------------------cleanup-------------------------------------*/
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_cpu_C);

    return EXIT_SUCCESS;
}
