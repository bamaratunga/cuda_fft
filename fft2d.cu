/***********************************************************
*
* Developed for Seminar in Parallelisation of Physics
* Calculations on GPUs with CUDA, Department of Physics
* Technical University of Munich.
*
* Author: Binu Amaratunga
*
*
***********************************************************/
#include <math.h>
#include <cuda.h>
#include "utils.h"
#include "controls.h"

/*************************************
* Generate reversed index to a given
* output
*
*
*
**************************************/
__device__ unsigned int bitReversed(unsigned int input, unsigned int Nbits){
  unsigned int rev = 0;
  for(int i = 0; i < Nbits; i++){
    rev <<= 1;
    if(input & 1 == 1)
      rev ^= 1;
    input >>= 1;
  }
  return rev;
}

/*************************************
* Kernel to re-arrange output with
* bit reversed order
*
*
*
**************************************/

__global__ void bitReversedCopy(double * reBuffer, double * imBuffer, int N, int N_stages){
    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    // If N_stages is odd, copy back to second half to synchronize
    if(N_stages % 2){
        reBuffer[kernelId_y * N + kernelId_x] = reBuffer[N * N + kernelId_y * N + kernelId_x];
        imBuffer[kernelId_y * N + kernelId_x] = imBuffer[N * N + kernelId_y * N + kernelId_x];
    }

    reBuffer[N * N + kernelId_y * N + kernelId_x]
                                = reBuffer[kernelId_y * N + bitReversed(kernelId_x, N_stages)];
    imBuffer[N * N + kernelId_y * N + kernelId_x]
                                = imBuffer[kernelId_y * N + bitReversed(kernelId_x, N_stages)];
}

/*************************************
* Transpose Input matrix
*
*
*
*
**************************************/

__global__ void transpose(double * reBuffer, double * imBuffer, int N){
    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    reBuffer[kernelId_y * N + kernelId_x] = reBuffer[N * N +  kernelId_x * N + kernelId_y];
    imBuffer[kernelId_y * N + kernelId_x] = imBuffer[N * N +  kernelId_x * N + kernelId_y];
}

/*************************************
* Get the absolute value for output
*
*
*
*
**************************************/

__global__ void calcAbsolute(double * reBuffer, double * imBuffer, int N){
    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    reBuffer[N * N + kernelId_y * N + kernelId_x]
            = reBuffer[kernelId_y * N + kernelId_x] * reBuffer[kernelId_y * N + kernelId_x]
            + imBuffer[kernelId_y * N + kernelId_x] * imBuffer[kernelId_y * N + kernelId_x];
}

/*************************************
* Kernel to calculate one instance
* of a single stage of fft
*
*
*
**************************************/

__global__ void fft_stage(double * reBuffer, double * imBuffer, int N, int stage, int N_stages){

    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Number of patitions
    unsigned int N_parts = 1 << stage;
    // Number of elements in a partition
    unsigned int N_elems = __double2uint_rn(N / N_parts);
    // Current partition Number
    int part =  __double2int_rn(kernelId_x / N_elems);
    // Element number in the current partition
    int elem = kernelId_x - part * N_elems;

    int mult_factor = ((part + 2) % 2 ) * (-1) + ((part + 1) % 2 );
    // Calculate respective sums
    double reSumValue =
          mult_factor * reBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + part * N_elems + elem]
        + reBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + ( part + mult_factor ) * N_elems + elem];

    double imSumValue =
          mult_factor * imBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + part * N_elems + elem]
        + imBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + ( part + mult_factor ) * N_elems + elem];

    double cos_t = cos(2.0 * M_PI * elem * ( 1 << (stage - 1)) / N );
    double sin_t = sin(2.0 * M_PI * elem * ( 1 << (stage - 1)) / N );

    // Calculate multiplication of sum with Wn
    double reMulValue = cos_t * reSumValue  + sin_t * imSumValue;
    double imMulValue = cos_t * imSumValue  - sin_t * reSumValue;

    // Do the selection - if to consider the multiplication factor or not
    reBuffer[(stage % 2) * N * N + kernelId_y * N + part * N_elems + elem] =
                                                              ((part + 2) % 2) * reMulValue
                                                            + ((part + 1) % 2) * reSumValue;

    imBuffer[(stage % 2) * N * N + kernelId_y * N + part * N_elems + elem] =
                                                              ((part + 2) % 2) * imMulValue
                                                            + ((part + 1) % 2) * imSumValue;
}

/*************************************
* Perform fft in 2D
*
*
*
*
**************************************/

void fft2(double * inputData, double * outputData, const unsigned int N) {

    double * d_reInputBuf = NULL;
    double * d_imInputBuf = NULL;

    // Allocate double the size of array to do double buffering
    gpuErrChk(cudaMalloc(&d_reInputBuf, 2 * N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imInputBuf, 2 * N * N * sizeof(double)));

    gpuErrChk(cudaMemcpy(d_reInputBuf, inputData, N * N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemset(d_imInputBuf, 0, N * N * sizeof(double)));

    // Number of stages in the FFT
    unsigned int N_stages = log(N) / log(2);
    // Resource allocation
    // TODO: Generalize this part
    unsigned int gSize = max(1, N/32);
    unsigned int bSize = min(N, 32);
    dim3 gridSize(gSize, gSize);
    dim3 blockSize( bSize, bSize); // Multiples of 32

    // Perform 1D FFT row wise and column wise
    for(int iter = 0; iter < 2; iter++){

        for(int stage = 1; stage < N_stages + 1; stage++){
            fft_stage<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N, stage, N_stages);
        }

        bitReversedCopy<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N, N_stages);

        // Transpose matrix
        transpose<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N);
    }

    calcAbsolute<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N);

    gpuErrChk(cudaDeviceSynchronize());
    gpuErrChk(cudaMemcpy(outputData, d_reInputBuf + N * N, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(d_reInputBuf));
    gpuErrChk(cudaFree(d_imInputBuf));
}
