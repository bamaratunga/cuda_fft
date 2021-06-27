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

void fft(double * reInput, double * imInput, double * reOutput, double * imOutput, int N, int step){
	if (N == 1) {
    reOutput[0] = reInput[0];
    imOutput[0] = imInput[0];
    return;
  }

  int half = N / 2;
  fft(reInput, imInput, reOutput, imOutput, half, 2 * step);
  fft(reInput + step, imInput + step, reOutput + half, imOutput + half, half, 2 * step);

  for (int k = 0; k < half; ++k) {
    double theta = - 2.0 * PI * (double)(k) / (double)(N);
    double rePart = cos(theta);
    double imPart = sin(theta);

    double reUpdate = reOutput[k + half] * rePart - imOutput[k + half] * imPart;
    double imUpdate = rePart * imOutput[k + half] + reOutput[k + half] * imPart;

    double reOrig = reOutput[k];
    double imOrig = imOutput[k];

    reOutput[k] = reOrig + reUpdate;
    imOutput[k] = imOrig + imUpdate;

    reOutput[k + half] = reOrig - reUpdate;
    imOutput[k + half] = imOrig - imUpdate;
  }
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
