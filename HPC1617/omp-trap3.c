/****************************************************************************
 *
 * omp-trap3.c - Parallel implementation of the trapezoid rule
 *
 * Original by Peter Pacheco http://www.cs.usfca.edu/~peter/ipp/
 * Modified in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -fopenmp -Wall -Wpedantic omp-trap3.c -o omp-trap3 -lgomp
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp_trap3
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
 * Function to be integrated
 */
double f( double x )
{
    return 4.0/(1.0 + x*x);
}

/*
 * Compute the area of function f(x) for x=[a, b] using the trapezoid
 * rule. The integration interval [a,b] is partitioned into n
 * subintervals of equal size.
 */
double trap( double a, double b, int n )
{
    double result = 0.0;
    const double h = (b-a)/n;
    int i;
    /* The loop index variable is automatically private, so we don't
       need to worry about it. h is constant, so there is no race
       condition. result is within a 'reduction' clause, so it is
       guaranteed to be updated correctly. */
#pragma omp parallel for reduction(+:result)
    for ( i = 0; i<n; i++ ) {
	result += h*(f(a+i*h) + f(a+(i+1)*h))/2;
    }
    return result;
}

int main( int argc, char* argv[] )
{
    double a = 0.0, b = 1.0, result;
    int n = 1000000;
    double tstart, tstop;

    if ( 4 == argc ) {
	a = atof(argv[1]);
	b = atof(argv[2]);
	n = atoi(argv[3]);
    }

    tstart = omp_get_wtime();
    result = trap(a, b, n);
    tstop = omp_get_wtime();

    printf("Area: %f\n", result);
    printf("Elapsed time %f\n", tstop - tstart);
    return 0;
}
