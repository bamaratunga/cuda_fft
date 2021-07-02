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
#include <string.h>
#include <math.h>
#include <time.h>

#include "utils.h"

// Transpose Input matrix
void transpose(double * input, double * output, int N){
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      output[j * N + i] = input[i * N + j];
      output[j * N + i] = input[i * N + j];
    }
  }
}

// Generate reversed index to a given output
unsigned int bitReversed(unsigned int input, unsigned int Nbits){
  unsigned int rev = 0;
  for(int i = 0; i < Nbits; i++){
    rev <<= 1;
    if(input & 1 == 1)
      rev ^= 1;
    input >>= 1;
  }
  return rev;
}

// Calculate FFT of a single vector
void fft(double * reInput, double * imInput, double * reOutput, double * imOutput, int N, int step){

  // Create buffers of twice the size to store data interchangeably
  double * reBuffer = (double *)malloc(2 * N * sizeof(double));
  double * imBuffer = (double *)malloc(2 * N * sizeof(double));

  // Initialize data
  for(int i = 0; i < N; i++){
    reBuffer[i] = reInput[i];
    imBuffer[i] = 0;
    reBuffer[N + i] = 0;
    imBuffer[N + i] = 0;
  }

  // Number of stages in the FFT
  unsigned int N_stages = log(N) / log(2);

  double reSumValue;
  double imSumValue;
  double reMulValue;
  double imMulValue;

  for(int stage = 1; stage < N_stages + 1; stage++){

    // Number of patitions
    unsigned int N_parts = pow(2, stage);
    // Number of elements in a partition
    unsigned int N_elems = N / N_parts;
    // Loop through each partition
    for(int part = 0; part < N_parts; part++){
      // Loop through each element in partition
      for(int elem = 0; elem < N_elems; elem++){

        // Calculate respective sums
        reSumValue = ( pow(-1, (part + 2) % 2) * reBuffer[((stage + 1) % 2) * N + part * N_elems + elem]
                   + reBuffer[((stage + 1) % 2) * N + ( part + (int)pow(-1, (part + 2) % 2) ) * N_elems + elem] );

        imSumValue = ( pow(-1, (part + 2) % 2) * imBuffer[((stage + 1) % 2) * N + part * N_elems + elem]
                   + imBuffer[((stage + 1) % 2) * N + ( part + (int)pow(-1, (part + 2) % 2) ) * N_elems + elem] );

        // Calculate multiplication of sum with Wn (Omega / e(j*pi*k / N))
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
    }
  }

  // Bit reversed indexed copy to rearrange in the correct order
  for(unsigned int i = 0; i < N; i++){
      reOutput[i] = reBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
      imOutput[i] = imBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
  }

  free(reBuffer);
  free(imBuffer);
}


void fft2(double * input, double * output, int N)
{
  double * reInput = (double *)malloc(N * N * sizeof(double));
  double * imInput = (double *)malloc(N * N * sizeof(double));

  double * reInter = (double *)malloc(N * N * sizeof(double));
  double * imInter = (double *)malloc(N * N * sizeof(double));

  // Copy input real data to input buffers
  memcpy(reInput, input, N * N * sizeof(double));

  // Perform row wise FFT
  for (int j = 0; j < N; j++){
      fft(&reInput[j * N], &imInput[j * N], &reInter[j * N], &imInter[j * N], N, 1);
  }
  // Transpose output and overwrite the input buffer
  transpose(reInter, reInput, N);
  transpose(imInter, imInput, N);

  // Perform FFT (This time column wise)
  for (int j = 0; j < N; j++){
    fft(&reInput[j * N], &imInput[j * N], &reInter[j * N], &imInter[j * N], N, 1);
  }
  // Transpose to get the original output
  transpose(reInter, reInput, N);
  transpose(imInter, imInput, N);

  // Calculate amplitude for the output
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      output[j * N + i] = reInput[j * N + i] * reInput[j * N + i] + imInput[j * N + i] * imInput[j * N + i];
    }
  }

  free(reInput);
  free(imInput);
  free(reInter);
  free(imInter);
}


int main(int argc, char** argv){

  if(argc < 2) {
    printf("Enter the dimension size as argument!\n");
    exit(EXIT_FAILURE);
  }

  int N = atoi(argv[1]);

  double * inputData = (double *)malloc(N * N * sizeof(double));
  double * amplitudeOut = (double *)malloc(N * N * sizeof(double));

  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      inputData[j * N + i] = 0.0;
      // Set slit positions to 1
      if ((abs(i-N/2) <= 10) && (abs(i-N/2) >= 8) && (abs(j-N/2) <= 4)){
        inputData[j * N + i] = 1.0;
      }
      // printf("%0.0lf ", reInput[j * N + i]);
    }
    // printf("\n");
  }

  printf("Running fft for %d x %d = %d = 2 ^ %d data points...\n", N, N, N*N, (int)(log(N*N)/log(2)));

  clock_t start, end;
  double cpu_time_used;

  start = clock();
	fft2(inputData, amplitudeOut, N);
  end = clock();

  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

  printf("Runtime = %lfs\n", cpu_time_used);

  printf("Writing output data...\n");
  writeCSV(amplitudeOut, 0, N);

	free(inputData);
  free(amplitudeOut);

	return 0;
}
