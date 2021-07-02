# Fourier Transform with CUDA

2 Dimensional double precision Fourier transform implementation with CUDA for NVIDIA GPUs using different approaches for benchmarking purposes.

  - Direct computation of Discrete Fourier Transform
  - Fast Fourier Transform implementation
  - Fast Fourier Transform with cuFFT library

Developed for Seminar in Parallelisation of Physics Calculations on GPUs with CUDA, Department of Physics, Technical University of Munich.

The application is to simulate the Young's double slit experiment with Fraunhofer Diffraction.

## Compilation and running

To compile the project:
```
make all
```

To run hand implemented FFT:
```
./fft2d <size of one dimension>
```

For cuFFT implementaion:
```
./cufft2d <size of one dimension>
```

For direct DFT implementation:
```
./fft2d <size of one dimension>
```

The C model for FFT can be compiled and run with:
```
make c
./fft2dc <size of one dimension>

```
## License
[MIT](https://choosealicense.com/licenses/mit/)
