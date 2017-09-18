/****************************************************************************
 *
 * reduction.c - Tree-structured reduction
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
 * gcc -std=c99 -Wall -pedantic reduction.c -o reduction
 *
 * Run with:
 * ./reduction
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* Serial implementation of the sum-reduce operator */
float sum_reduce_serial( float* x, size_t n )
{
    float result = 0.0;
    int i;
    for (i=0; i<n; i++) {
        result += x[i];
    }
    return result;
}

/* Tree-structured implementation of the sum-reduce operator; this
   implementation works for any vector length n > 0 */
float sum_reduce( float* x, size_t n )
{
    int d, i;

    /* compute the largest power of two < n */
    for (d=1; 2*d < n; d *= 2) 
        ;
    /* do reduction */
    for ( ; d>0; d /= 2) {
        for (i=0; i<d; i++) {
            if (i+d<n) x[i] += x[i+d];
        }
    }
    return x[0];
}

int main( void )
{
    float v[51], result;
    int n = sizeof(v) / sizeof(v[0]);
    int i;
    for (i=0; i<n; i++) {
	v[i] = i+1;
    }
    result = sum_reduce(v, n);
    if ( fabs(result - n*(n+1)/2.0) > 1e-5 ) {
        printf("Error: expected %f, got %f\n", n*(n+1)/2.0, result);
        return -1;
    }
    printf("Test ok\n");
    return 0;
}
