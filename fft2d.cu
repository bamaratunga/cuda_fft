#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#define PI 3.14159265

int N = 32;

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

// Transpose Input matrix
__global__ void transpose(double * input, double * output, int N){
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      output[j * N + i] = input[i * N + j];
      output[j * N + i] = input[i * N + j];
    }
  }
}

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
               + reBuffer[((stage + 1) % 2) * N + ( part + (int)pow(-1, (part + 2) % 2) ) * N_elems + elem] );

    imSumValue = ( pow(-1, (part + 2) % 2) * imBuffer[((stage + 1) % 2) * N + part * N_elems + elem]
               + imBuffer[((stage + 1) % 2) * N + ( part + (int)pow(-1, (part + 2) % 2) ) * N_elems + elem] );

    // Calculate multiplication of sum with Wn
    reMulValue = cos(2.0 * PI * elem * pow(2, (stage - 1)) / N ) * reSumValue
               + sin(2.0 * PI * elem * pow(2, (stage - 1)) / N ) * imSumValue;
    imMulValue = cos(2.0 * PI * elem * pow(2, (stage - 1)) / N ) * imSumValue
               - sin(2.0 * PI * elem * pow(2, (stage - 1)) / N ) * reSumValue;

    // Do the selection - if to consider the multiplication factor or not
    reBuffer[(stage % 2) * N + part * N_elems + elem] =
                                    ((part + 2) % 2) * reMulValue
                                  + ((part + 1) % 2) * reSumValue;

    imBuffer[(stage % 2) * N + part * N_elems + elem] =
                                    ((part + 2) % 2) * imMulValue
                                  + ((part + 1) % 2) * imSumValue;
}


// Calculate FFT of a single vector
                    /* Input buffers -> size = 2*N        |   Output buffers -> size = N     */
__global__ void fft(double * reBuffer, double * imBuffer, double * reOutput, double * imOutput, int N){

    // Number of stages in the FFT
    unsigned int N_stages = log(N) / log(2);

    dim3 gridSize(N / 32, N / 32);
    dim3 blockSize( 32, 32); // Multiples of 32

    for(int stage = 1; stage < N_stages + 1; stage++){
        gpuErrChk(fft_stage<<<gridSize, blockSize>>>(d_reBuffer, d_imBuffer, N, stage));
    }

    // Bit reversed indexed copy to rearrange in the correct order
    for(unsigned int i = 0; i < N; i++){
      reOutput[i] = reBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
      imOutput[i] = imBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
    }
}


__host__ void fft2(double * inputData, double * outputData, int N)
{
    double * d_reInputBuf = NULL;
    double * d_imInputBuf = NULL;

    gpuErrChk(cudaMalloc(&d_reInputBuf, N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imInputBuf, N * N * sizeof(double)));

    gpuErrChk(cudaMemcpy(d_reInputBuf, inputData, N * N * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrChk(cudaMemset(d_imInputBuf, 0, N * N * sizeof(double)));

    // Number of stages in the FFT
    unsigned int N_stages = log(N) / log(2);
    // Resource allocation
    dim3 gridSize(N / 32, N / 32);
    dim3 blockSize( 32, 32); // Multiples of 32

    double * d_reBuffer = NULL;
    double * d_imBuffer = NULL;

    gpuErrChk(cudaMalloc(&d_reBuffer, 2 * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_imBuffer, 2 * N * sizeof(double)));

    // Peform FFT to each row
    for(int j = 0; j < N; j++){

        // TODO: Separate kernel to do copying faster?
        // TODO:                         // Check for correctness
        gpuErrChk(cudaMemcpy(d_reBuffer, d_reInputBuf + N * j, N * sizeof(double), cudaMemcpyDeviceToDevice));
        gpuErrChk(cudaMemcpy(d_imBuffer, d_imInputBuf + N * j, N * sizeof(double), cudaMemcpyDeviceToDevice));

        for(int stage = 1; stage < N_stages + 1; stage++){
            gpuErrChk(fft_stage<<<gridSize, blockSize>>>(d_reBuffer, d_imBuffer, N, stage));
        }

        gpuErrChk(cudaDeviceSynchronize());

        gpuErrChk(cudaMemcpy(d_reInputBuf + N * j, d_reBuffer + N * (stage % 2), N * sizeof(double), cudaMemcpyDeviceToDevice));
        gpuErrChk(cudaMemcpy(d_imInputBuf + N * j, d_imBuffer + N * (stage % 2), N * sizeof(double), cudaMemcpyDeviceToDevice));
    }



  double * reBuffer = (double *)malloc(2 * N * sizeof(double));
  double * imBuffer = (double *)malloc(2 * N * sizeof(double));

  double * reTemp = (double *)malloc(N * N * sizeof(double));
  double * imTemp = (double *)malloc(N * N * sizeof(double));

  reBuffer[i] = reInput[i];
  imBuffer[i] = imInput[i];
  reBuffer[N + i] = 0;
  imBuffer[N + i] = 0;

  // memcpy(reBuffer, reInput, N * sizeof(double));
  // memcpy(imBuffer, imInput, N * sizeof(double));



  fft(reInput, imInput, reTemp, imTemp, reBuffer, imBuffer, N);

  double * reTemp2 = (double *)malloc(N * N * sizeof(double));
  double * imTemp2 = (double *)malloc(N * N * sizeof(double));

  transpose(reTemp, reTemp2, N);
  transpose(imTemp, imTemp2, N);


  fft(reTemp2, imTemp2, reTemp, imTemp, reBuffer, imBuffer, N);


  transpose(reTemp, reOutput, N);
  transpose(imTemp, imOutput, N);

  free(reTemp);
  free(imTemp);
}


int main()
{
  double * inputData = (double *)malloc(N * N * sizeof(double));
  double * outputData = (double *)malloc(N * N * sizeof(double));

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
