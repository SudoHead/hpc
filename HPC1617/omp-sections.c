/*
 * omp_sections.c - Demostration of the OpenMP "sections" work sharing directive
 *
 * Compile with:
 * gcc -fopenmp omp_sections.c -o omp_sections
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp_sections
 *
 * Author: Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified: 2013/11/02
 * License: CC0 1.0 Public Domain Dedication 
 *          http://creativecommons.org/publicdomain/zero/1.0/
 */
#include <stdio.h>
#define N 1000

int main(void)
{
    int i;
    float a[N], b[N], c[N], d[N];
    /* Some initializations */
    for (i=0; i < N; i++) {
	a[i] = i * 1.5;
	b[i] = i + 22.35;
    }
#pragma omp parallel shared(a,b,c,d) private(i)
    {
#pragma omp sections nowait
	{
#pragma omp section
	    for (i=0; i < N; i++)
		c[i] = a[i] + b[i];
#pragma omp section
	    for (i=0; i < N; i++)
		d[i] = a[i] * b[i];
	}  /* end of sections */
    }  /* end of parallel section */

    for ( i=0; i<N; i++) {
	printf("%f %f %f %f\n", a[i], b[i], c[i], d[i]);
    }

    return 0;
}
