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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>

#include "controls.h"
#include "utils.h"

/*************************************
* Compute 2D FFT with cuFFT
* output
*
*
*
**************************************/

void  fft2(cuDoubleComplex * inData, const unsigned int N) {

	cufftDoubleComplex *d_inData = NULL;

	gpuErrChk(cudaMalloc(&d_inData, N * N * sizeof(cufftDoubleComplex)));

    gpuErrChk(cudaMemcpy(d_inData, inData, N * N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

	cufftHandle plan;
    cufftResult flag;

	flag = cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
    if ( CUFFT_SUCCESS != flag ) printf("2D: cufftPlan2d fails!\n");

	flag = cufftExecZ2Z(plan, d_inData, d_inData, CUFFT_FORWARD);
	if ( CUFFT_SUCCESS != flag ) printf("2D: cufftExecR2C fails!\n");

    gpuErrChk(cudaDeviceSynchronize());
	gpuErrChk(cudaMemcpy(inData, d_inData, N * N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost) );

	flag = cufftDestroy(plan);
    if ( CUFFT_SUCCESS != flag ) printf("2D: cufftDestroy fails!\n");
	gpuErrChk(cudaFree(d_inData));
}


int main(int argc, char** argv){

  if(argc < 2) {
      printf("Enter the dimension size as argument!\n");
      exit(EXIT_FAILURE);
  }

  int N = atoi(argv[1]);

  // Complex data input
  cuDoubleComplex * inputData = (cuDoubleComplex *)malloc(N * N * sizeof(cuDoubleComplex));
  // Real data
  double * outputData = (double *)malloc(N * N * sizeof(double));

  int slit_height = 1;
  int slit_width  = 2;
  int slit_dist   = 7;

  // TODO: Create this data on the device itself
  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      inputData[j * N + i] = make_cuDoubleComplex(0.0, 0.0);
      // Set slit positions to 1
      if ((abs(i-N/2) <= slit_dist+slit_width) && (abs(i-N/2) >= slit_dist) && (abs(j-N/2) <= slit_height)){
        inputData[j * N + i] = make_cuDoubleComplex(1.0, 0.0);
      } // printf("%0.0lf ", reInput[j * N + i]);
    } // printf("\n");
  }

  printf("Running fft for %d x %d = %d = 2 ^ %d data points...\n", N, N, N*N, (int)(log(N*N)/log(2)));

  clock_t start, end;
  double cpu_time_used;

  start = clock();
  fft2(inputData, N);
  // TODO: Do this in cuBLAS
  for(int i = 0; i < N*N; i++){
      outputData[i] = cuCreal(inputData[i]) * cuCreal(inputData[i])
                    + cuCimag(inputData[i]) * cuCimag(inputData[i]);
  }
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Runtime = %lfs\n", cpu_time_used);

  printf("Writing output data...\n");
  writeCSV(outputData, 0, N);

  free(inputData);
  free(outputData);

  return 0;
}
