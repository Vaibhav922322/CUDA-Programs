// nvcc -g -G -arch=sm_61 -std=c++11 22111064-prob1.cu -o 22111064-prob1
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)

#define SIZE1 8196
#define SIZE2 8200
#define ITER 100

#define BLOCKDIM 512
#define GRIDDIM ( SIZE1 / BLOCKDIM )
#define NUMTHREADS (BLOCKDIM*GRIDDIM)

using std::cerr;
using std::cout;
using std::endl;

__global__ void kernel1(const double* d_k1_in, double* d_k1_out) {
    // TODO: Fill in
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    for (int k = 0; k < ITER; k++)
    {
        for (int i = 1; i < (SIZE1 - 1); i++)
        {
            for (int j = threadId; j < (SIZE1 - 1); j += NUMTHREADS)
            {
                d_k1_out[i * SIZE1 + j + 1] =
                    (
                        d_k1_in[(i - 1) * SIZE1 + j + 1] +
                        d_k1_in[i * SIZE1 + j + 1] +
                        d_k1_in[(i + 1) * SIZE1 + j + 1]
                        );
            }
        }
    }
}

__global__ void kernel2(double* d_k2_in, double* d_k2_out) {
    // TODO: Fill in
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    for (int k = 0; k < ITER; k++)
    {
        for (int i = 1; i < (SIZE2 - 1); i++)
        {
            d_k2_out[i * SIZE2 + threadId + 1] =
                (
                    d_k2_in[(i - 1) * SIZE2 + threadId + 1] +
                    d_k2_in[i * SIZE2 + threadId + 1] +
                    d_k2_in[(i + 1) * SIZE2 + threadId + 1]
                    );
        }
    }
    for (int k = 0; k < ITER; k++)
    {
        for (int i = threadId + 1; i < (SIZE2 - 1); i += NUMTHREADS)
        {
            for (int j = NUMTHREADS;j < (SIZE2 - 1); j++)
            {
                d_k2_out[i * SIZE2 + j + 1] =
                    (
                        d_k2_in[(i - 1) * SIZE2 + j + 1] +
                        d_k2_in[i * SIZE2 + j + 1] +
                        d_k2_in[(i + 1) * SIZE2 + j + 1]
                        );
            }
        }
    }

}

__host__ void serial(const double* h_ser_in, double* h_ser_out) {
    for (int k = 0; k < ITER; k++) {
        for (int i = 1; i < (SIZE1 - 1); i++) {
            for (int j = 0; j < (SIZE1 - 1); j++) {
                h_ser_out[i * SIZE1 + j + 1] =
                    (h_ser_in[(i - 1) * SIZE1 + j + 1] + h_ser_in[i * SIZE1 + j + 1] +
                        h_ser_in[(i + 1) * SIZE1 + j + 1]);
            }
        }
    }
}
__host__ void serial2(const double* h_ser_in, double* h_ser_out) {
    for (int k = 0; k < ITER; k++) {
        for (int i = 1; i < (SIZE2 - 1); i++) {
            for (int j = 0; j < (SIZE2 - 1); j++) {
                h_ser_out[i * SIZE2 + j + 1] =
                    (h_ser_in[(i - 1) * SIZE2 + j + 1] + h_ser_in[i * SIZE2 + j + 1] +
                        h_ser_in[(i + 1) * SIZE2 + j + 1]);
            }
        }
    }
}

__host__ void check_result(const double* w_ref, const double* w_opt,
    const uint64_t size) {
    double maxdiff = 0.0;
    int numdiffs = 0;

    for (uint64_t i = 0; i < size; i++) {
        for (uint64_t j = 0; j < size; j++) {
            double this_diff = w_ref[i * size + j] - w_opt[i * size + j];
            if (fabs(this_diff) > THRESHOLD) {
                numdiffs++;
                if (this_diff > maxdiff)
                    maxdiff = this_diff;
            }
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
            << "; Max Diff = " << maxdiff << endl;
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
    double* h_ser_in = new double[SIZE1 * SIZE1];
    double* h_ser_out = new double[SIZE1 * SIZE1];
    double* h_ser_in2 = new double[SIZE2 * SIZE2];
    double* h_ser_out2 = new double[SIZE2 * SIZE2];

    double* h_k1_in = new double[SIZE1 * SIZE1];
    double* h_k1_out = new double[SIZE1 * SIZE1];

    for (int i = 0; i < SIZE1; i++) {
        for (int j = 0; j < SIZE1; j++) {
            h_ser_in[i * SIZE1 + j] = 1;
            h_ser_out[i * SIZE1 + j] = 1;
            h_k1_in[i * SIZE1 + j] = 1;
            h_k1_out[i * SIZE1 + j] = 1;
        }
    }

    double* h_k2_in = new double[SIZE2 * SIZE2];
    double* h_k2_out = new double[SIZE2 * SIZE2];

    for (int i = 0; i < SIZE2; i++) {
        for (int j = 0; j < SIZE2; j++) {
            h_k2_in[i * SIZE2 + j] = 1;
            h_k2_out[i * SIZE2 + j] = 1;
            h_ser_in2[i * SIZE2 + j] = 1;
            h_ser_out2[i * SIZE2 + j] = 1;
        }
    }

    double clkbegin = rtclock();
    serial(h_ser_in, h_ser_out);
    double clkend = rtclock();
    double time = clkend - clkbegin; // seconds
    cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
        << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

    cudaError_t status;
    cudaEvent_t start, end;
    float k1_time; // ms

    double* d_k1_in;
    double* d_k1_out;

    // TODO: FILL IN, USE EVENT TIMERS
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //Allocating memory into GPU
    cudaMalloc(&d_k1_in, sizeof(double) * SIZE1 * SIZE1);
    cudaMalloc(&d_k1_out, sizeof(double) * SIZE1 * SIZE1);

    // Marking start time
    cudaEventRecord(start);

    // Copy data from CPU to GPU 
    status = cudaMemcpy(d_k1_in, h_k1_in, sizeof(double) * SIZE1 * SIZE1, cudaMemcpyHostToDevice);
    status = cudaMemcpy(d_k1_out, h_k1_out, sizeof(double) * SIZE1 * SIZE1, cudaMemcpyHostToDevice);

    if (status != cudaSuccess) {
        cerr << cudaGetErrorString(status) << endl;
    }

    // Launching Kernal1 Method
    kernel1 << < GRIDDIM, BLOCKDIM >> > (d_k1_in, d_k1_out);

    // Waiting for Kernal1 to finish
    cudaDeviceSynchronize();

    // Copy data from GPU to CPU
    cudaMemcpy(h_k1_out, d_k1_out, sizeof(double) * SIZE1 * SIZE1, cudaMemcpyDeviceToHost);

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Calculating time spend by Kernal1 Method
    cudaEventElapsedTime(&k1_time, start, end);

    // Comparing the results
    check_result(h_ser_out, h_k1_out, SIZE1);
    cout << "Kernel 1 on GPU: "
        << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
        << " GFLOPS; Time = " << k1_time << " msec" << endl;


    serial2(h_ser_in2, h_ser_out2);

    double* d_k2_in;
    double* d_k2_out;
    // TODO: FILL IN, USE EVENT TIMERS
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //Allocating memory into GPU
    cudaMalloc(&d_k2_in, sizeof(double) * SIZE2 * SIZE2);
    cudaMalloc(&d_k2_out, sizeof(double) * SIZE2 * SIZE2);

    // Marking start time
    cudaEventRecord(start);

    // Copy data from CPU to GPU 
    cudaMemcpy(d_k2_in, h_k2_in, sizeof(double) * SIZE2 * SIZE2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k2_out, h_k2_out, sizeof(double) * SIZE2 * SIZE2, cudaMemcpyHostToDevice);

    // Launching Kernal2 Method
    kernel2 << < GRIDDIM, BLOCKDIM >> > (d_k2_in, d_k2_out);

    // Waiting for Kernal2 to finish
    cudaDeviceSynchronize();

    // Copy data from GPU to CPU 
    cudaMemcpy(h_k2_out, d_k2_out, sizeof(double) * SIZE2 * SIZE2, cudaMemcpyDeviceToHost);

    // Marking end time 
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Calculating time spend by Kernal2 Method
    cudaEventElapsedTime(&k1_time, start, end);
    check_result(h_ser_out2, h_k2_out, SIZE2);

    cout << "Kernel 2 on GPU: "
        << ((2.0 * SIZE2 * SIZE2 * ITER) / (k1_time * 1.0e-3))
        << " GFLOPS; Time = " << k1_time << " msec" << endl;

    cudaFree(d_k1_in);
    cudaFree(d_k1_out);
    cudaFree(d_k2_in);
    cudaFree(d_k2_out);

    delete[] h_ser_in;
    delete[] h_ser_out;
    delete[] h_ser_in2;
    delete[] h_ser_out2;


    delete[] h_k1_in;
    delete[] h_k1_out;


    delete[] h_k2_in;
    delete[] h_k2_out;

    return EXIT_SUCCESS;
}

