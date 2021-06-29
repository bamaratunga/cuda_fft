#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

const int N = 32;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// Write ouput to CSV file
__host__ void writeCSV(double * input, int idx){
  char fname[0x100];
  snprintf(fname, sizeof(fname), "output_%d.csv", idx);
  FILE *fp = fopen(fname, "w");

  for(int col = 0; col < N; col++){
    for(int row = 0; row < N-1; row++){
      fprintf(fp, "%lf, ", input[row + N * col]);
    }
    fprintf(fp, "%lf", input[N-1 + N * col]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

// Generate reversed index to a given output
__device__ unsigned int bitReversed(unsigned int input, int Nbits){
  unsigned int rev = 0;
  for(int i = 0; i < Nbits; i++){
    rev <<= 1;
    if(input & 1 == 1)
      rev ^= 1;
    input >>= 1;
  }
  return rev;
}

// Kernel to re-arrange output with bit reversed order
__global__ void bitReversedCopy(double * reBuffer, double * imBuffer, int N, int N_stages){
    // Max idx is N
    unsigned int kernel_idx = blockIdx.x*blockDim.x + threadIdx.x;

    reBuffer[((N_stages + 1) % 2) * N + kernel_idx] = reBuffer[(N_stages % 2) * N + bitReversed(kernel_idx, N_stages)];
    imBuffer[((N_stages + 1) % 2) * N + kernel_idx] = imBuffer[(N_stages % 2) * N + bitReversed(kernel_idx, N_stages)];
}

// Transpose Input matrix
__global__ void transpose(double * input, double * output, int N){
    // Max idx is N
    unsigned int kernel_idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < N; j++){
        output[j * N + kernel_idx] = input[kernel_idx * N + j];
    }
}

// Get the absolute value for output
__global__ void calcAbsolute(double * reInput, double * imInput, int N){
    // Max idx is N
    unsigned int kernel_idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int j = 0; j < N; j++){
        reInput[j * N + kernel_idx] *= reInput[kernel_idx * N + j];
        imInput[j * N + kernel_idx] *= imInput[kernel_idx * N + j];
        reInput[j * N + kernel_idx] += imInput[kernel_idx * N + j];
    }
}

// Kernel to calculate one instance of a single stage
__global__ void fft_stage(double * reBuffer, double * imBuffer, int N, int stage){

    // Max idx is N
    unsigned int kernel_idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Number of patitions
    unsigned int N_parts = pow(2, stage);
    // Number of elements in a partition
    unsigned int N_elems = N / N_parts;
    // Current partition Number
    unsigned int part = kernel_idx / N_parts;
    // Element number in the current partition
    unsigned int elem = kernel_idx - part * N_elems;

    // Temporary variables
    double reSumValue;
    double imSumValue;
    double reMulValue;
    double imMulValue;

    // Calculate respective sums
    reSumValue = ( pow(-1, (part + 2) % 2) * reBuffer[((stage + 1) % 2) * N + part * N_elems + elem]
               + reBuffer[((stage + 1) % 2) * N + ( part + __double2uint_rn(pow(-1, (part + 2) % 2)) ) * N_elems + elem] );

    imSumValue = ( pow(-1, (part + 2) % 2) * imBuffer[((stage + 1) % 2) * N + part * N_elems + elem]
               + imBuffer[((stage + 1) % 2) * N + ( part + __double2uint_rn(pow(-1, (part + 2) % 2)) ) * N_elems + elem] );

    // Calculate multiplication of sum with Wn
    reMulValue = cos(2.0 * M_PI * elem * pow(2, (stage - 1)) / N ) * reSumValue
               + sin(2.0 * M_PI * elem * pow(2, (stage - 1)) / N ) * imSumValue;
    imMulValue = cos(2.0 * M_PI * elem * pow(2, (stage - 1)) / N ) * imSumValue
               - sin(2.0 * M_PI * elem * pow(2, (stage - 1)) / N ) * reSumValue;

    // Do the selection - if to consider the multiplication factor or not
    reBuffer[(stage % 2) * N + part * N_elems + elem] =
                                    ((part + 2) % 2) * reMulValue
                                  + ((part + 1) % 2) * reSumValue;

    imBuffer[(stage % 2) * N + part * N_elems + elem] =
                                    ((part + 2) % 2) * imMulValue
                                  + ((part + 1) % 2) * imSumValue;
}

// Perform fft in 2D
__host__ void fft2(double * inputData, double * outputData, int N)
{
    double * d_reInputBuf = NULL;
    double * d_imInputBuf = NULL;
    double * d_reInputBuf_T = NULL;
    double * d_imInputBuf_T = NULL;
    double * d_reTemp = NULL;
    double * d_imTemp = NULL;

    gpuErrChk(cudaMalloc(&d_reInputBuf, N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imInputBuf, N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_reInputBuf_T, N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imInputBuf_T, N * N * sizeof(double)));

    gpuErrChk(cudaMemcpy(d_reInputBuf, inputData, N * N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemset(d_imInputBuf, 0, N * N * sizeof(double)));

    // Number of stages in the FFT
    unsigned int N_stages = log(N) / log(2);
    // Resource allocation
    dim3 gridSize(N / 32, N / 32);
    dim3 blockSize( 32, 32); // Multiples of 32

    double * d_reBuffer = NULL;
    double * d_imBuffer = NULL;

    // 2 * N size buffers to do double buffering between stages
    gpuErrChk(cudaMalloc(&d_reBuffer, 2 * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imBuffer, 2 * N * sizeof(double)));

    for(int iteration = 0; iteration < 2; iteration++){
        // Peform FFT to each row
        for(int j = 0; j < N; j++){

            // TODO: Separate kernel to do copying faster?
            // TODO:                         // Check for correctness
            gpuErrChk(cudaMemcpy(d_reBuffer, d_reInputBuf + N * j, N * sizeof(double), cudaMemcpyDeviceToDevice));
            gpuErrChk(cudaMemcpy(d_imBuffer, d_imInputBuf + N * j, N * sizeof(double), cudaMemcpyDeviceToDevice));

            for(int stage = 1; stage < N_stages + 1; stage++){
                fft_stage<<<gridSize, blockSize>>>(d_reBuffer, d_imBuffer, N, stage);
                gpuErrChk(cudaDeviceSynchronize());
            }

            gpuErrChk(cudaDeviceSynchronize());

            bitReversedCopy<<<gridSize, blockSize>>>(d_reBuffer, d_imBuffer, N, N_stages);

            gpuErrChk(cudaDeviceSynchronize());

            gpuErrChk(cudaMemcpy(d_reInputBuf + N * j, d_reBuffer + N * ((N_stages + 1) % 2), N * sizeof(double), cudaMemcpyDeviceToDevice));
            gpuErrChk(cudaMemcpy(d_imInputBuf + N * j, d_imBuffer + N * ((N_stages + 1) % 2), N * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // Transpose matrix
        transpose<<<gridSize, blockSize>>>(d_reInputBuf, d_reInputBuf_T, N);
        transpose<<<gridSize, blockSize>>>(d_imInputBuf, d_imInputBuf_T, N);

        // Swap pointers
        d_reTemp = d_reInputBuf;
        d_imTemp = d_imInputBuf;
        d_reInputBuf = d_reInputBuf_T;
        d_imInputBuf = d_imInputBuf_T;
        d_reInputBuf_T = d_reTemp;
        d_imInputBuf_T = d_imTemp;
    }

    calcAbsolute<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N);

    gpuErrChk(cudaMemcpy(outputData, d_reInputBuf, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(d_reInputBuf));
    gpuErrChk(cudaFree(d_imInputBuf));
    gpuErrChk(cudaFree(d_reBuffer));
    gpuErrChk(cudaFree(d_imBuffer));
}


int main()
{
  double * inputData = (double *)malloc(N * N * sizeof(double));
  double * outputData = (double *)malloc(N * N * sizeof(double));

  // TODO: Create this data on the device itself
  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      inputData[j * N + i] = 0.0;
      // Set slit positions to 1
      if ((abs(i-N/2) <= 10) && (abs(i-N/2) >= 8) && (abs(j-N/2) <= 4)){
        inputData[j * N + i] = 1.0;
      } // printf("%0.0lf ", reInput[j * N + i]);
    } // printf("\n");
  }

  fft2(inputData, outputData, N);

  writeCSV(outputData, 0);

  free(inputData);
  free(outputData);

  return 0;
}
