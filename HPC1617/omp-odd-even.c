/****************************************************************************
 *
 * omp-odd-even.c - Odd-even transposition sort with OpenMP
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * OpenMP implementation of odd-even transposition sort.
 *
 * To compile:
 * gcc -fopenmp -std=c99 -Wall -pedantic omp-odd-even.c -o omp-odd-even -lgomp
 *
 * Run with:
 * OMP_NUM_THREADS=4 ./omp-odd-even
 *
 ***************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
    }
}

/* Fills vector v with a permutation of the integer values 0, .. n-1 */
void fill( int* v, int n )
{
    int i;
    int up = n-1, down = 0;
    for ( i=0; i<n; i++ ) {
	v[i] = ( i % 2 == 0 ? up-- : down++ );
    }
}

/* Odd-even transposition sort; this function uses two omp parallel
   for directives. */
void odd_even_sort_nopool( int* v, int n )
{
    int phase, i;
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
#pragma omp parallel for default(none) private(i) shared(n,v)
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
#pragma omp parallel for default(none) private(i) shared(n,v)
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

/* Same as above, but with a common pool of threads that are recycled
   in the omp for constructs */
void odd_even_sort_pool( int* v, int n )
{
    int phase, i;

#pragma omp parallel default(none) private(i,phase) shared(n,v)
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
#pragma omp for
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
#pragma omp for
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

void check( int* v, int n )
{
    int i;
    for (i=0; i<n-1; i++) {
	if ( v[i] != i ) {
	    printf("Check failed: v[%d]=%d, expected %d\n",
		   i, v[i], i );
	    abort();
	}
    }
    printf("Check ok!\n");
}

int main( int argc, char* argv[] )
{
    int n = 100000;
    int *v;
    int r;
    const int NREPS = 5;
    double tstart, tstop;

    if ( argc > 1 ) {
	n = atoi(argv[1]);
    }
    v = (int*)malloc(n*sizeof(v[0]));
    fill(v,n);
    printf("Without thread pool recycling: \t");
    tstart = hpc_gettime();
    for (r=0; r<NREPS; r++) {        
        odd_even_sort_nopool(v,n);
    }
    tstop = hpc_gettime();
    printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
    check(v,n);
    fill(v,n);
    printf("With thread pool recycling: \t");
    tstart = hpc_gettime();
    for (r=0; r<NREPS; r++) {
        odd_even_sort_pool(v,n);
    }
    tstop = hpc_gettime();
    printf("Mean elapsed time %f\n", (tstop - tstart)/NREPS);
    check(v,n);
    return 0;
}
