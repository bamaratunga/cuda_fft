#include <stdio.h>
#include <math.h>
#include <complex.h>

double PI;

void _fft(complex buf[], complex out[], int n, int step)
{
	if (step < n) {
		_fft(out, buf, n, step * 2);
		_fft(out + step, buf + step, n, step * 2);

		for (int i = 0; i < n; i += 2 * step) {
			complex t = cexp(-I * PI * i / n) * out[i + step];
			buf[i / 2]     = out[i] + t;
			buf[(i + n)/2] = out[i] - t;
		}
	}
}

void fft(complex buf[], int n)
{
	complex out[n];
	for (int i = 0; i < n; i++) out[i] = buf[i];

	_fft(buf, out, n, 1);
}


void show(const char * s, complex buf[]) {
	printf("%s", s);
	for (int i = 0; i < 8; i++)
		if (!cimag(buf[i]))
			printf("%g ", creal(buf[i]));
		else
			printf("(%g, %g) ", creal(buf[i]), cimag(buf[i]));
}

int main()
{
	PI = atan2(1, 1) * 4;
	complex buf[] = {1, 1, 1, 1, 0, 0, 0, 0};

	show("Data: ", buf);
	fft(buf, 8);
	show("\nFFT : ", buf);

	return 0;
}
