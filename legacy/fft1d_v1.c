#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265

int N = 8;

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

  printf("\nInput: \n");
  for(int i = 0; i < N; i++){
    printf("(%lf, %lf) \n", reInput[i], imInput[i]);
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
