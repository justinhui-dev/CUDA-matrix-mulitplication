#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"
#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32 // 16

__global__ void gemm_gpu_kernel(float *a, float *b, float *result, int dim) {

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int i,j;
    float temp = 0;

    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;

        A_shared[threadIdx.y][threadIdx.x] = a[row * dim + j];
        B_shared[threadIdx.y][threadIdx.x] = b[i * dim + col]; 

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            temp += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x]; 
        }
        __syncthreads();
    }

    result[row * dim + col] = temp;
}

__host__ int init_matrix(float **L_mat, float **R_mat, int L_dimX, int L_dimY, int R_dimX, int R_dimY) {

    int sqr_dim_X = R_dimX;
    if (L_dimX > R_dimX) {
        sqr_dim_X = L_dimX;
    }

    int sqr_dim_Y = R_dimY;
    if (L_dimY > R_dimY) {
        sqr_dim_Y = L_dimY;
    }

    int size = sqr_dim_Y;
    if (sqr_dim_X > sqr_dim_Y) {
        size = sqr_dim_X;
    }

    int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
    size = temp * BLOCK_SIZE;

    size_t pt_size = size * size * sizeof(float);

    *L_mat = (float *) malloc(pt_size);
    *R_mat = (float *) malloc(pt_size);

    memset(*L_mat, 0, pt_size);
    memset(*R_mat, 0, pt_size);

    for (int i = 0; i < L_dimX; i++) {
        for (int j = 0; j < L_dimY; j++) {
            int temp1 = int(size * i + j);
            (*L_mat)[temp1] = sinf(temp1);
        }
    }
    for (int i = 0; i < R_dimX; i++) {
        for (int j = 0; j < R_dimY; j++) {
            int temp2 = int(size * i + j);
            (*R_mat)[temp2] = cosf(temp2);
        }
    }
    return size;
}

__host__ void gemm_cpu(float *a, float *b, float *result, int m) {

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            float tmp = 0.0;

            for (int h = 0; h < m; ++h) {
                tmp += a[i * m + h] * b[h * m + j];
            }
            result[i * m + j] = tmp;
        }
    }
}

int main(void) {

    int a_dimX, a_dimY;
    int b_dimX, b_dimY;

    float *a_cpu, *b_cpu;
    float *a_gpu, *b_gpu;

    float *Result_cpu;
    float *Result_gpu;
    float *Result_Final;

    a_dimX = 1024;
    a_dimY = 1024;
    b_dimX = 1024;
    b_dimY = 1024;

    int dim = init_matrix(&a_cpu, &b_cpu, a_dimX, a_dimY, b_dimX, b_dimY);

    size_t vector_size;
    vector_size = dim*dim * sizeof(float);

    Result_cpu = (float *) malloc(vector_size);
    Result_Final = (float *) malloc(vector_size);

    cudaMalloc((void **) &a_gpu, vector_size);
    cudaMalloc((void **) &b_gpu, vector_size);
    cudaMalloc((void **) &Result_gpu, vector_size);

    cudaMemcpy(a_gpu, a_cpu, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, vector_size, cudaMemcpyHostToDevice);

    dim3 gridDim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Grid: %dx%d blocks\n", gridDim.x, gridDim.y);
    printf("Blocks: %dx%d threads\n", blockDim.x, blockDim.y);
    printf("Shared Mem: %d bytes\n", int(sizeof(float) * 2 * BLOCK_SIZE * BLOCK_SIZE));
    printf("Matrix A: %dx%d\n", a_dimX, a_dimY);
    printf("Matrix B: %dx%d\n", b_dimX, b_dimY);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    gemm_gpu_kernel <<< gridDim, blockDim >>> (a_gpu, b_gpu, Result_gpu, dim);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float gpu_t;
    cudaEventElapsedTime(&gpu_t, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(Result_cpu, Result_gpu, vector_size, cudaMemcpyDeviceToHost);

    clock_t start_t = clock();

    gemm_cpu(a_cpu,b_cpu,Result_Final,dim);

    clock_t finish_t = clock();
    double cpu_t = (double)1000*(finish_t - start_t) / CLOCKS_PER_SEC;


    printf("GPU time: %f ms\n", gpu_t);
    printf("CPU time: %lf ms\n", cpu_t);

    bool flag = true;
    for (int i=0;i< a_dimX && flag;i++){
        for (int j = 0; j < b_dimY && flag; j++) {
            if (abs(Result_cpu[i*dim+j] - Result_Final[i*dim+j]) > 0.0001) {
                flag = false;
                printf("%f != %f\n", Result_cpu[i*dim+j], Result_Final[i*dim+j]);
            }
        }
    }

    if (flag) {
        printf("Success!\n");
    }

    free(a_cpu);
    free(b_cpu);
    free(Result_cpu);
    free(Result_Final);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(Result_gpu);
}