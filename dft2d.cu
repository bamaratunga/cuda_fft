#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#include<math.h>
#include<cuda.h>

// Size of grid
// const int N = 32;

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


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


__global__ void fft(double * inputData, double * amplitudeOut, int N){
    double realOut = 0;
    double imagOut = 0;

    int xWave = blockIdx.x*blockDim.x + threadIdx.x;
    int yWave = blockIdx.y*blockDim.y + threadIdx.y;

    int height = N;
    int width = N;

    for (int ySpace = 0; ySpace < height; ySpace++) {
        for (int xSpace = 0; xSpace < width; xSpace++) {
            // Compute real, imag, and ampltude.
            realOut += (inputData[ySpace * width + xSpace] * cos(2.0 * M_PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height))));
            imagOut -= (inputData[ySpace * width + xSpace] * sin(2.0 * M_PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height))));
        }
    }
    // amplitudeOut[yWave * n + xWave] = sqrt(realOut * realOut + imagOut * imagOut);
    amplitudeOut[yWave * N + xWave] = (realOut * realOut + imagOut * imagOut);
}


int main(int argc, char **argv) {

    if(argc < 2) {
        printf("Enter the dimension size as argument!\n");
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    int i, j;

    double * inputData = (double *)malloc(N * N * sizeof(double));
    double * amplitudeOut = (double *)malloc(N * N * sizeof(double));

    // TODO: Create this data on the device itself
    for (j = 0; j < N; j++){
        for (i = 0; i < N; i++){
            inputData[j*N + i] = 0.0;
            // Set slit positions to 1
            if ((abs(i-N/2) <= 10) && (abs(i-N/2) >= 8) && (abs(j-N/2) <= 4)){
              inputData[j*N + i] = 1.0;
            }
            amplitudeOut[j*N + i] = 0.0;
        }
    }

    clock_t start, end;
    double cpu_time_used;

    printf("Running fft for %d x %d = %d = 2 ^ %d data points...\n", N, N, N*N, (int)(log(N*N)/log(2)));

    start = clock();

    double * d_inputData = NULL;
    double * d_amplitudeOut = NULL;

    gpuErrChk(cudaMalloc(&d_inputData, N * N * sizeof(double)));
    gpuErrChk(cudaMalloc(&d_amplitudeOut, N * N * sizeof(double)));

    gpuErrChk(cudaMemcpy(d_inputData, inputData, N * N * sizeof(double), cudaMemcpyHostToDevice));

    dim3 gridSize(N / 32, N / 32);
    dim3 blockSize( 32, 32); // Multiples of 32

    fft<<<gridSize, blockSize>>>(d_inputData, d_amplitudeOut, N);

    gpuErrChk(cudaDeviceSynchronize());

    gpuErrChk(cudaMemcpy(amplitudeOut, d_amplitudeOut, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Runtime = %lfs\n", cpu_time_used);

    printf("Writing output data...\n");
    writeCSV(amplitudeOut, 0, N);

    gpuErrChk(cudaFree(d_inputData));
    gpuErrChk(cudaFree(d_amplitudeOut));

    free(inputData);
    free(amplitudeOut);

    return 0;
}
