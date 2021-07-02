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
#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>

/*******************************
* Writes ouput to CSV file
*
*
*******************************/
inline void writeCSV(double * input, int idx, unsigned int N){

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

#endif // _UTILS_H_
