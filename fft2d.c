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
#include <math.h>
#include <time.h>

#include "utils.h"
#include "fft2d.h"

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
