#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

#define PI 3.14159265

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Size of grid
int N = 32;

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
            realOut += (inputData[ySpace * width + xSpace] * cos(2.0 * PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height)))) / sqrt(1.0 * width * height);
            imagOut -= (inputData[ySpace * width + xSpace] * sin(2.0 * PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height)))) / sqrt(1.0 * width * height);
        }
    }
    // amplitudeOut[yWave * n + xWave] = sqrt(realOut * realOut + imagOut * imagOut);
    amplitudeOut[yWave * N + xWave] = sqrt(realOut * realOut + imagOut * imagOut);
}



int main(int argc, char **argv) {

    int i, j;

    double * inputData = new double[N * N]();
    double * amplitudeOut = new double[N * N]();

    for (j = 0; j < N; j++){
        for (i = 0; i < N; i++){
            inputData[j*N + i] = 0.0;
            // Set slit positions to 1
            if ((abs(i-N/2) <= 10) && (abs(i-N/2) >= 8) && (abs(j-N/2) <= 4)){
              inputData[j*N + i] = 1.0;
            }
        }
    }

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

    for (j = 0; j < N; j++){
        for (i = 0; i < N; i++){
            printf("%lf ", N * N * amplitudeOut[j * N + i] * amplitudeOut[j * N + i]);
        }
        printf("\n");
    }

    return 0;
}
