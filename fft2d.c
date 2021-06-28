#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265

int N = 32;

void transpose(double * input, double * output, int N){

  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      output[j * N + i] = input[i * N + j];
      output[j * N + i] = input[i * N + j];
    }
  }
}

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

void fft(double * reInput, double * imInput, double * reOutput, double * imOutput, int N, int step){

  double * reBuffer = (double *)malloc(2 * N * sizeof(double));
  double * imBuffer = (double *)malloc(2 * N * sizeof(double));

  for(int i = 0; i < N; i++){
    reBuffer[i] = reInput[i];
    imBuffer[i] = imInput[i];
    reBuffer[N + i] = 0;
    imBuffer[N + i] = 0;
  }
  // memcpy(reBuffer, reInput, N * sizeof(double));
  // memcpy(imBuffer, imInput, N * sizeof(double));

  unsigned int N_stages = log(N) / log(2);
  double reSumValue;
  double imSumValue;
  double reMulValue;
  double imMulValue;

  for(int stage = 1; stage < N_stages + 1; stage++){

    unsigned int N_parts = pow(2, stage);
    unsigned int N_elems = N / N_parts;
    for(int part = 0; part < N_parts; part++){

      for(int elem = 0; elem < N_elems; elem++){

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
        // Do the selection
        reBuffer[(stage % 2) * N + part * N_elems + elem] =
                                        ((part + 2) % 2) * reMulValue
                                      + ((part + 1) % 2) * reSumValue;

        imBuffer[(stage % 2) * N + part * N_elems + elem] =
                                        ((part + 2) % 2) * imMulValue
                                      + ((part + 1) % 2) * imSumValue;
      }
    }
  }

  // Bit reversed copy
  for(unsigned int i = 0; i < N; i++){
      reOutput[i] = reBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
      imOutput[i] = imBuffer[(N_stages % 2) * N + bitReversed(i, N_stages)];
  }

  free(reBuffer);
  free(imBuffer);
}


void fft2(double * reInput, double * imInput, double * reOutput, double * imOutput, int N)
{
  double * reTemp = (double *)malloc(N * N * sizeof(double));
  double * imTemp = (double *)malloc(N * N * sizeof(double));

  for (int j = 0; j < N; j++){
      fft(&reInput[j * N], &imInput[j * N], &reTemp[j * N], &imTemp[j * N], N, 1);
  }

  double * reTemp2 = (double *)malloc(N * N * sizeof(double));
  double * imTemp2 = (double *)malloc(N * N * sizeof(double));

  transpose(reTemp, reTemp2, N);
  transpose(imTemp, imTemp2, N);

  for (int j = 0; j < N; j++){
    fft(&reTemp2[j * N], &imTemp2[j * N], &reTemp[j * N], &imTemp[j * N], N, 1);
  }

  transpose(reTemp, reOutput, N);
  transpose(imTemp, imOutput, N);

  free(reTemp);
  free(imTemp);
  free(reTemp2);
  free(imTemp2);
}


int main()
{
  double * reInput = (double *)malloc(N * N * sizeof(double));
  double * imInput = (double *)malloc(N * N * sizeof(double));

  double * reOutput = (double *)malloc(N * N * sizeof(double));
  double * imOutput = (double *)malloc(N * N * sizeof(double));

  // Create double slit
  for (int j = 0; j < N; j++){
    for (int i = 0; i < N; i++){
      reInput[j * N + i] = 0.0;
      imInput[j * N + i] = 0.0;
      // Set slit positions to 1
      if ((abs(i-N/2) <= 10) && (abs(i-N/2) >= 8) && (abs(j-N/2) <= 4)){
        reInput[j * N + i] = 1.0;
        imInput[j * N + i] = 0.0;
      }
      // printf("%0.0lf ", reInput[j * N + i]);
    }
    // printf("\n");
  }

	fft2(reInput, imInput, reOutput, imOutput, N);

  for(int j = 0; j < N; j++){
    for(int i = 0; i < N; i++){
      printf("(%lf, %lf) ", reOutput[j * N + i], imOutput[j * N + i]);
    }
    printf("\n");
  }

	free(reInput);
  free(imInput);
	free(reOutput);
  free(imOutput);

	return 0;
}
