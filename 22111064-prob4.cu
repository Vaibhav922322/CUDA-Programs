// nvcc -g -G -arch=sm_61 -std=c++11 22111064-prob4.cu -o 22111064-prob4

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (64);
#define THRESHOLD (0.000001)
#define BLOCKDIM 8

using std::cerr;
using std::cout;
using std::endl;

// TODO: Edit the function definition as required
__global__ void kernel1(const double* d_in, double* d_out)
{
    uint64_t matrix = N * N;
    int i = blockIdx.y + blockIdx.z * gridDim.z;
    int j = threadIdx.z + blockDim.x * blockIdx.y;
    int k = threadIdx.y * blockDim.x + threadIdx.x;

    // i,j,k should be starting from 1 not 0
    i++;
    j++;
    k++;

    if (i < N - 1 && j < N - 1 && k < N - 1)
    {
        d_out[i * matrix + j * N + k] = 0.8 * (
            d_in[(i - 1) * matrix + j * N + k] + d_in[(i + 1) * matrix + j * N + k] +
            d_in[i * matrix + (j - 1) * N + k] + d_in[i * matrix + (j + 1) * N + k] +
            d_in[i * matrix + j * N + (k - 1)] + d_in[i * matrix + j * N + (k + 1)]);
    }
}

// TODO: Edit the function definition as required
__global__ void kernel2(const double* d_in, double* d_out)
{
    uint64_t matrix = N * N;
    const uint16_t sharedMemDim = BLOCKDIM + 2;
    const uint64_t matrixShared = sharedMemDim * sharedMemDim;

    __shared__ uint64_t s_in[matrixShared * sharedMemDim];
    __shared__ uint64_t s_out[matrixShared * sharedMemDim];

    int i = blockDim.z * blockIdx.z * threadIdx.z;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    uint16_t tx = threadIdx.x + 1;
    uint16_t ty = threadIdx.y + 1;
    uint16_t tz = threadIdx.z + 1;

    if ( i < N  && j < N && k < N )
    {
        // copying data to shared memory
        s_in[tz  * matrixShared + ty * sharedMemDim + tx ] = d_in[i * matrix + j * N + k];
        s_out[tz  * matrixShared + ty * sharedMemDim + tx ] = d_out[i * matrix + j * N + k];

        if (tz == 1 && i!= 0)
        {
            s_in[ (tz-1) * matrixShared + ty * sharedMemDim + tx] = d_in[ (i-1) * matrix + j * N + k];
            s_out[ (tz-1) * matrixShared + ty * sharedMemDim + tx] = d_out[ (i-1) * matrix + j * N + k];
        }
        if (tz == BLOCKDIM && i < N - 1)
        {
            s_in[(tz + 1) * matrixShared + ty * sharedMemDim + tx] = d_in[(i + 1) * matrix + j * N + k];
            s_out[(tz + 1) * matrixShared + ty * sharedMemDim + tx] = d_out[(i + 1) * matrix + j * N + k];
        }

        if (tx == 1 && k != 0)
        {
            s_in[tz * matrixShared + ty * sharedMemDim + tx - 1] = d_in[i * matrix + j * N + k - 1];
            s_out[tz * matrixShared + ty * sharedMemDim + tx - 1] = d_out[i * matrix + j * N + k - 1];
        }
        if (tx == BLOCKDIM && k < N - 1)
        {
            s_in[tz * matrixShared + ty * sharedMemDim + tx + 1] = d_in[i * matrix + j * N + k + 1];
            s_out[tz * matrixShared + ty * sharedMemDim + tx + 1] = d_out[i * matrix + j * N + k + 1];
        }
        
        if (ty == 1 && j != 0)
        {
            s_in[tz * matrixShared + (ty-1) * sharedMemDim + tx] = d_in[i * matrix + (j-1) * N + k ];
            s_out[tz * matrixShared + (ty-1) * sharedMemDim + tx] = d_out[i * matrix + (j-1) * N + k ];
        }
        if (ty == BLOCKDIM && j < N - 1)
        {
            s_in[tz * matrixShared + ( ty + 1 ) * sharedMemDim + tx] = d_in[i * matrix + (j + 1) * N + k];
            s_out[tz * matrixShared + ( ty + 1 ) * sharedMemDim + tx] = d_out[i * matrix + (j + 1) * N + k];           
        }
        // Every single threads must come to this point before continue
        __syncthreads();

        // i,j,k should be starting from 1 not 0
        if (i > 0 && j > 0 && k > 0 && i < N - 1  && j < N - 1 && k < N - 1)
        {
            s_out[tz * matrixShared + ty * sharedMemDim + tx] = 0.8 * (
                    s_in[(tz - 1) * matrixShared + ty * sharedMemDim + tx] + s_in[(tz + 1) * matrixShared + ty * sharedMemDim + tx] +
                    s_in[tz * matrixShared + (ty - 1) * sharedMemDim + tx] + s_in[tz * matrixShared + (ty + 1) * sharedMemDim + tx] +
                    s_in[tz * matrixShared + ty * sharedMemDim + (tx - 1)] + s_in[tz * matrixShared + ty * sharedMemDim + (tx + 1)]);
            __syncthreads();
            d_out[i * matrix + j * N + k] = s_out[tz * matrixShared + ty * sharedMemDim + tx];    
        }
    }
}

// TODO: Edit the function definition as required
__host__ void stencil(const double* h_in, double* h_out)
{
    uint64_t matrix = N * N;
    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            for (int k = 1; k < N - 1; k++)
            {
                h_out[i * matrix + j * N + k] = 0.8 * (
                    h_in[(i - 1) * matrix + j * N + k] + h_in[(i + 1) * matrix + j * N + k] +
                    h_in[i * matrix + (j - 1) * N + k] + h_in[i * matrix + (j + 1) * N + k] +
                    h_in[i * matrix + j * N + (k - 1)] + h_in[i * matrix + j * N + (k + 1)]);
            }
        }
    }
}

__host__ void check_result(const double* w_ref, const double* w_opt, const uint64_t size) {
    double maxdiff = 0.0, this_diff = 0.0;
    int numdiffs = 0;

    for (uint64_t i = 0; i < size; i++) {
        for (uint64_t j = 0; j < size; j++) {
            for (uint64_t k = 0; k < size; k++) {
                this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
                if (std::fabs(this_diff) > THRESHOLD) {
                    numdiffs++;
                    if (this_diff > maxdiff) {
                        maxdiff = this_diff;
                    }
                }
            }
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
            << endl;
    }
    else {
        cout << "No differences found between base and test versions\n";
    }
}

__host__ void print_mat(double* A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                printf("%lf,", A[i * N * N + j * N + k]);
            }
            printf("      ");
        }
        printf("\n");
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
    uint64_t SIZE = N * N * N;
    double* h_in, * h_out, * h_GPU_out;

    h_in = (double*)malloc(SIZE * sizeof(double));
    h_out = (double*)malloc(SIZE * sizeof(double));
    h_GPU_out = (double*)malloc(SIZE * sizeof(double));

    // Initialize Matrix
    for (uint64_t i = 0; i < N; i++)
    {
        for (uint64_t j = 0; j < N; j++)
        {
            for (uint64_t k = 0; k < N; k++)
            {
                h_in[(i * N * N) + (j * N) + k] = rand() % 64;
                h_out[(i * N * N) + (j * N) + k] = rand() % 64;
            }
        }
    }

    // Executing serial method (stencil)
    double clkbegin = rtclock();
    stencil(h_in, h_out);
    double clkend = rtclock();
    double cpu_time = clkend - clkbegin;
    cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

    cudaError_t status;
    cudaEvent_t start, end;

    double* d_in, * d_out;

    // TODO: Fill in kernel1
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Defining and initializing grid and block dimensions
    dim3 block(BLOCKDIM, BLOCKDIM, BLOCKDIM);
    int blockPerGrid = ceil(N / BLOCKDIM);
    dim3 grid(N / BLOCKDIM, BLOCKDIM, BLOCKDIM);

    //Allocating memory on GPU
    status = cudaMalloc(&d_in, SIZE * sizeof(double));
    if (status != cudaSuccess) {
        cerr << cudaGetErrorString(status) << endl;
    }
    cudaMalloc(&d_out, SIZE * sizeof(double));

    // Copying data from CPU to GPU
    cudaMemcpy(d_in, h_in, SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Marking start time
    cudaEventRecord(start);

    // launching Kernal1
    kernel1 << < grid, block >> > (d_in, d_out);
    cudaDeviceSynchronize();

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Copying data from GPU to CPU
    cudaMemcpy(h_GPU_out, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // TODO: Adapt check_result() and invoke
    check_result(h_out, h_GPU_out, N);
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";
    double *h_GPU2_out = (double*)malloc(SIZE * sizeof(double));

    // TODO: Fill in kernel2

    // Marking start time
    cudaEventRecord(start);

    // launching Kernal2
    kernel2 << < grid, block >> > (d_in, d_out);
    cudaDeviceSynchronize();

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Copying data from GPU to CPU
    status = cudaMemcpy(h_GPU2_out, d_out, SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cerr << cudaGetErrorString(status) << endl;
    }

    // TODO: Adapt check_result() and invoke
    check_result(h_out, h_GPU_out, N);
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";


    // TODO: Free memory
    cudaFree(d_out);
    cudaFree(d_in);

    free(h_in);
    free(h_out);
    free(h_GPU_out);
    free(h_GPU2_out);
    return EXIT_SUCCESS;
}