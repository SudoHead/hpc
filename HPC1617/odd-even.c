/****************************************************************************
 *
 * odd-even.c - Serial implementation of the odd-even transposition sort.
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
 * Compile with:
 * gcc -std=c99 -Wall -pedantic odd-even.c -o odd-even
 *
 * Run with:
 * ./odd-even 1000
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

/* Fills vector v with a permutation of the integer values 0, .. n-1 */
void fill( int* v, int n )
{
    int i;
    int up = n-1, down = 0;
    for ( i=0; i<n; i++ ) {
	v[i] = ( i % 2 == 0 ? up-- : down++ );
    }
}

/* Odd-even transposition sort */
void odd_even_sort( int* v, int n )
{
    int phase, i;
    double tstart = hpc_gettime();
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
    double tstop = hpc_gettime();
    printf("Sorting time %f\n", tstop - tstart);
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
    int n = 50000;
    int* v;

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }
    v = (int*)malloc(n*sizeof(v[0]));
    fill(v,n);
    odd_even_sort(v,n);
    check(v,n);
    return 0;
}
