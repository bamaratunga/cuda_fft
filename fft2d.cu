#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<cuda.h>

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
__host__ void writeCSV(double * input, int idx, unsigned int N){
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

// Transpose Input matrix
__global__ void transpose(double * reBuffer, double * imBuffer, int N){
    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    reBuffer[kernelId_y * N + kernelId_x] = reBuffer[N * N +  kernelId_x * N + kernelId_y];
    imBuffer[kernelId_y * N + kernelId_x] = imBuffer[N * N +  kernelId_x * N + kernelId_y];
}

// Get the absolute value for output
__global__ void calcAbsolute(double * reBuffer, double * imBuffer, int N){
    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    reBuffer[N * N + kernelId_y * N + kernelId_x]
            = reBuffer[kernelId_y * N + kernelId_x] * reBuffer[kernelId_y * N + kernelId_x]
            + imBuffer[kernelId_y * N + kernelId_x] * imBuffer[kernelId_y * N + kernelId_x];
}

// Kernel to calculate one instance of a single stage
__global__ void fft_stage(double * reBuffer, double * imBuffer, int N, int stage, int N_stages){

    // Max idx is N
    unsigned int kernelId_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int kernelId_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Number of patitions
    unsigned int N_parts = __double2uint_rn(pow(2, stage));
    // Number of elements in a partition
    unsigned int N_elems = __double2uint_rn(N / N_parts);
    // Current partition Number
    unsigned int part =  __double2uint_rn(kernelId_x / N_elems);
    // Element number in the current partition
    int elem = kernelId_x - part * N_elems;

    // Temporary variables
    double reSumValue;
    double imSumValue;
    double reMulValue;
    double imMulValue;

    // Calculate respective sums
    reSumValue =
    __double2int_rn(pow(-1, (part + 2) % 2)) * reBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + part * N_elems + elem]
    + reBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + ( part + __double2int_rn(pow(-1, (part + 2) % 2)) ) * N_elems + elem];

    imSumValue =
    __double2int_rn(pow(-1, (part + 2) % 2)) * imBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + part * N_elems + elem]
    + imBuffer[((stage + 1) % 2) * N * N + kernelId_y * N + ( part + __double2int_rn(pow(-1, (part + 2) % 2)) ) * N_elems + elem];

    // Calculate multiplication of sum with Wn
    reMulValue = cos(2.0 * M_PI * elem * __double2uint_rn(pow(2, (stage - 1))) / N ) * reSumValue
               + sin(2.0 * M_PI * elem * __double2uint_rn(pow(2, (stage - 1))) / N ) * imSumValue;
    imMulValue = cos(2.0 * M_PI * elem * __double2uint_rn(pow(2, (stage - 1))) / N ) * imSumValue
               - sin(2.0 * M_PI * elem * __double2uint_rn(pow(2, (stage - 1))) / N ) * reSumValue;

    // Do the selection - if to consider the multiplication factor or not
    reBuffer[(stage % 2) * N * N + kernelId_y * N + part * N_elems + elem] =
                                                              ((part + 2) % 2) * reMulValue
                                                            + ((part + 1) % 2) * reSumValue;

    imBuffer[(stage % 2) * N * N + kernelId_y * N + part * N_elems + elem] =
                                                              ((part + 2) % 2) * imMulValue
                                                            + ((part + 1) % 2) * imSumValue;
}

// Perform fft in 2D
__host__ void fft2(double * inputData, double * outputData, const unsigned int N)
{
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
            // TODO: Could be unnecessary. Consider removing
            // gpuErrChk(cudaDeviceSynchronize());
        }
        // TODO: Could be unnecessary. Consider removing
        // gpuErrChk(cudaDeviceSynchronize());

        bitReversedCopy<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N, N_stages);

        // TODO: Could be unnecessary. Consider removing
        // gpuErrChk(cudaDeviceSynchronize());

        // Transpose matrix
        transpose<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N);
    }

    calcAbsolute<<<gridSize, blockSize>>>(d_reInputBuf, d_imInputBuf, N);

    gpuErrChk(cudaMemcpy(outputData, d_reInputBuf + N * N, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrChk(cudaFree(d_reInputBuf));
    gpuErrChk(cudaFree(d_imInputBuf));
}


int main(int argc, char** argv){

  if(argc < 2) {
    printf("Enter the dimension size as argument!\n");
    exit(EXIT_FAILURE);
  }

  int N = atoi(argv[1]);

  double * inputData = (double *)malloc(N * N * sizeof(double));
  double * outputData = (double *)malloc(N * N * sizeof(double));

  int slit_height = 4;
  int slit_width  = 2;
  int slit_dist   = 8;

  // TODO: Create this data on the device itself
  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      inputData[j * N + i] = 0.0;
      // Set slit positions to 1
      if ((abs(i-N/2) <= slit_dist+slit_width) && (abs(i-N/2) >= slit_dist) && (abs(j-N/2) <= slit_height)){
        inputData[j * N + i] = 1.0;
      } // printf("%0.0lf ", reInput[j * N + i]);
    } // printf("\n");
  }

  printf("Running fft for %d x %d = %d = 2 ^ %d data points...\n", N, N, N*N, (int)(log(N*N)/log(2)));

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  fft2(inputData, outputData, N);
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Runtime = %lfs\n", cpu_time_used);

  printf("Writing output data...\n");
  writeCSV(outputData, 0, N);

  free(inputData);
  free(outputData);

  return 0;
}
