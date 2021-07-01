#include<stdio.h>
#include<stdlib.h>

#include<cuda.h>
#include<cufft.h>
#include<cufftXt.h>
#include<cuComplex.h>

const int N = 4096;

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

// forward FFT (inplace)
// real data are put in contiguous data array, input[1:Nx, 1:Ny]
// but size of input is bigger, say  Nx * 2*(Ny>>1 +1) doublereal

// output:
// input is a complex array with size Nx*(Ny>>1 + 1)

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


int main()
{
  // Complex data input
  cuDoubleComplex * inputData = (cuDoubleComplex *)malloc(N * N * sizeof(cuDoubleComplex));
  // Real data
  double * outputData = (double *)malloc(N * N * sizeof(double));

  int slit_height = 4;
  int slit_width  = 2;
  int slit_dist   = 8;

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

  fft2(inputData, N);

  for(int i = 0; i < N*N; i++){
      outputData[i] = cuCreal(inputData[i]) * cuCreal(inputData[i])
                    + cuCimag(inputData[i]) * cuCimag(inputData[i]);
  }

  writeCSV(outputData, 0);

  free(inputData);
  free(outputData);

  return 0;
}
