/****************************************************************************
 *
 * omp-inclusive-scan.c - Implementation of the inclusive scan primitive
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
 * This file contains a serial implementation of the inclusive scan
 * primitive. Your goal is to parallelize this function using OpenMP,
 * according to the instructions given in the exercise sheet.
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

/* Fill v[] with the constant 1 */
void fill(int* v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = 1;
    }
}

/* Compute the inclusive scan of the n-elements array v[], and store
   the scan in s[]. The caller is responsible for allocating s with n
   elements */
void serial_inclusive_scan(int *v, int n, int *s)
{
    int i;
    /* degenerate case of empty array: do nothing */
    if ( n == 0 )
        return;

    s[0] = v[0];
    for (i=1; i<n; i++) {
        s[i] = s[i-1] + v[i];
    }
}

void check(int *s, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( s[i] != i+1 ) {
            printf("Check failed: expected s[%d]==%d, got %d\n", i, i+1, s[i]);
            abort();
        }
    }
    printf("Check ok!\n");
}

int main( int argc, char *argv[] )
{
    int n = 1000;
    int *v, *s;

    if ( argc > 2 ) {
        printf("Usage: %s [n]\n", argv[0]);
        return -1;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    v = (int*)malloc(n*sizeof(int));
    s = (int*)malloc(n*sizeof(int));
    fill(v, n);
    serial_inclusive_scan(v, n, s);
    check(s, n);
    free(v);
    free(s);
    return 0;
}
