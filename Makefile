all: dft2d fft2d cufft2d

c: fft2dc

cufft2d: cufft2d.cu
	nvcc -o cufft2d cufft2d.cu -O3 -lcufft

fft2d: fft2dc.o fft2dcu.o
	g++ fft2dc.o fft2dcu.o -o fft2d -lm -lcuda -lcudart

fft2dc.o: fft2d.c
	g++ -c -o fft2dc.o fft2d.c -O3

fft2dcu.o: fft2d.cu
	nvcc -c -o fft2dcu.o fft2d.cu -O3

fft2dc: fft2d_c.c
	g++ -o fft2dc fft2d_c.c -lm -O3

dft2d: dft2d.cu
	nvcc -o dft2d dft2d.cu

plot:
	python plot.py

clean:
	rm -f *.o
	rm -f cufft2d fft2d fft2dc dft2d
	rm -f *.csv

.PHONY: all clean plot
