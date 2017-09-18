/****************************************************************************
 *
 * prefix-sum.c - Tree-structured prefix-sum (inclusive scan) computation
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -std=c99 -Wall -pedantic prefix-sum.c -o prefix-sum
 *
 * Run with:
 * ./prefix-sum
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Prints the content of vector x[]; useful for debugging purposes */
void print( float* x, int n)
{
    int i;
    for (i=0; i<n; i++) {
        printf("[%d]=%f ", i, x[i]);
    }
    printf("\n");
}

/* Store in x[] the exclusive prefix-sum of x[] (i.e., at the end of
   this function x[0] == 0). NOTE: this function requires that n is a
   power of two. */
void prefix_sum( float* x, int n )
{
    int d, k;

#ifndef NDEBUG
    /* check whether n is a power of two */
    for (d=1; d<n; d <<= 1)
        ;
    assert(d == n);
#endif
    /* Phase 1: up-sweep */
    for ( d = 1; d<n/2; d <<= 1 ) {
	for ( k=0; k<n; k += 2*d ) {
            assert( k+2*d-1 < n );
	    x[k+2*d-1] = x[k+d-1] + x[k+2*d-1];
	}
    }

    /* Phase 2: down-sweep */
    x[n-1] = 0;
    for ( ; d > 0; d >>= 1 ) {
	for (k=0; k<n; k += 2*d ) {
            assert( k+2*d-1 < n );
	    float t = x[k+d-1];
	    x[k+d-1] = x[k+2*d-1];
	    x[k+2*d-1] = t + x[k+2*d-1];
	}
    }
}

int main( void )
{
    float v[32];
    int n = sizeof(v) / sizeof(v[0]);
    int i;
    for (i=0; i<n; i++) {
	v[i] = 1;
    }
    prefix_sum(v, n);
    for (i=0; i<n; i++) {
	if ( fabs(v[i] - i) > 1e-5 ) {
            printf("Error at index %d: expected %f, got %f\n", i, (double)i, v[i]);
            return -1;
        }
    }
    printf("Test ok\n");
    return 0;
}
