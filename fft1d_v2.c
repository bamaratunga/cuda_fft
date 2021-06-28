#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PI 3.14159265

int N = 32;

unsigned int bitReversed(unsigned int input, int Nbits){

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

int main()
{
  double * reInput = (double *)malloc(N * sizeof(double));
  double * imInput = (double *)malloc(N * sizeof(double));

  double * reOutput = (double *)malloc(N * sizeof(double));
  double * imOutput = (double *)malloc(N * sizeof(double));

  for(int i = 0; i < N; i++){
    reInput[i] = 0.0;
    imInput[i] = 0.0;
  }

  for(int i = 0; i < N/2; i++){
    reInput[i] = 1.0;
  }

  fft(reInput, imInput, reOutput, imOutput, N, 1);


  printf("\nOutput: \n");
  for(int i = 0; i < N; i++){
    printf("(%lf, %lf) \n", reOutput[i], imOutput[i]);
  }


  free(reInput);
  free(imInput);
  free(reOutput);
  free(imOutput);

  return 0;
}
