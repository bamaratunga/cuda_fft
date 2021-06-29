#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define PI 3.14159265

// Size of grid
int n = 32;

void writeCSV(double input[n][n], int idx){

  char fname[0x100];
  snprintf(fname, sizeof(fname), "output_%d.csv", idx);
  FILE *fp = fopen(fname, "w");

  for(int col = 0; col < n; col++){
    for(int row = 0; row < n-1; row++){
      fprintf(fp, "%lf, ", input[row][col]);
    }
    fprintf(fp, "%lf", input[n-1][col]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}


int main(int argc, char **argv) {

    double realOut[n][n];
    double imagOut[n][n];
    double amplitudeOut[n][n];

    int height = n;
    int width = n;
    int yWave;
    int xWave;
    int ySpace;
    int xSpace;

    int i, j;

    double inputData[n][n];


    for (j = 0; j < n; j++){
        for (i = 0; i < n; i++){
            inputData[i][j] = 0.0;
            // Set slit positions to 1
            if ((abs(i-n/2) <= 10) && (abs(i-n/2) >= 8) && (abs(j-n/2) <= 4)){
              inputData[i][j] = 1.0;
            }
            realOut[i][j] = 0;
            imagOut[i][j] = 0;
            amplitudeOut[i][j] = 0;
            // printf("%0.0lf ", inputData[i][j]);
        }
        // printf("\n");
    }

    // Two outer loops iterate on output data.
    for (xWave = 0; xWave < width; xWave++) {
        for (yWave = 0; yWave < height; yWave++) {
            //Two inner loops iterate on input data.
            for (ySpace = 0; ySpace < height; ySpace++) {
                for (xSpace = 0; xSpace < width; xSpace++) {
                    // Compute real, imag, and ampltude.
                    realOut[yWave][xWave] += (inputData[ySpace][xSpace] * cos(2 * PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height))));
                    imagOut[yWave][xWave] -= (inputData[ySpace][xSpace] * sin(2 * PI * ((1.0 * xWave * xSpace / width) + (1.0 * yWave * ySpace / height))));
                }
            }
            amplitudeOut[yWave][xWave] = (realOut[yWave][xWave] * realOut[yWave][xWave] + imagOut[yWave][xWave] * imagOut[yWave][xWave]);
            // printf("%f ", amplitudeOut[yWave][xWave]);
        }
        // printf("\n");
    }

    writeCSV(amplitudeOut, 0);

    return 0;
}
