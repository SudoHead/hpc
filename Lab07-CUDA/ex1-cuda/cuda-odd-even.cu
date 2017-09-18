/****************************************************************************
 *
 * cuda-odd-even.cu - Odd-even transposition sort with CUDA
 *
 * Written in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-odd-even.cu -o cuda-odd-even
 *
 * Run with:
 * ./cuda-odd-even [len]
 *
 * Example:
 * ./cuda-odd-even
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

/* if *a > *b, swap them. Otherwise do nothing */
void cmp_and_swap( int* a, int* b )
{
    if ( *a > *b ) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
    }
}

/* Odd-even transposition sort */
void odd_even_sort( int* v, int n )
{
    int phase, i;
    for (phase = 0; phase < n; phase++) {
	if ( phase % 2 == 0 ) {
	    /* (even, odd) comparisons */
	    for (i=0; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	} else {
	    /* (odd, even) comparisons */
	    for (i=1; i<n-1; i += 2 ) {
		cmp_and_swap( &v[i], &v[i+1] );
	    }
	}
    }
}

/**
 * Return a random integer in the range [a..b]
 */
int randab(int a, int b)
{
    return a + (rand() % (b-a+1));
}

/**
 * Fill vector x with a random permutation of the integers 0..n-1
 */
void fill( int *x, int n )
{
    int i, j, tmp;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
    for(i=0; i<n-1; i++) {
        j = randab(i+1, n-1);
        tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

int check( int *x, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != i) {
            printf("Check failed: x[%d]=%d, expected %d\n", i, x[i], i);
            return 0;
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char *argv[] ) 
{
    int *x;
    int n;
    const int default_len = 128*1024;
    double tstart, tend;

    if ( argc > 2 ) {
        printf("\nUsage: %s [len]\n\nSort an array of length \"len\" (default %d) using odd-even transposition sort\n\n", argv[0], default_len);
        return -1;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    } else {
        n = default_len;
    }

    const size_t size = n * sizeof(*x);

    /* Allocate space for host copies of x */
    x = (int*)malloc(size);
    fill(x, n);

    tstart = hpc_gettime();
    odd_even_sort(x, n);
    tend = hpc_gettime();
    printf("Elapsed %f\n", tend - tstart);

    /* Check result */
    check(x, n);

    /* Cleanup */
    free(x);
    return 0;
}
