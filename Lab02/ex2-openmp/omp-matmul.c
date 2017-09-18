/****************************************************************************
 *
 * omp-matmul.c - Cache-efficient dense matrix-matrix multiply application.
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
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-matmul.c -o omp-matmul
 *
 * Run with:
 * ./omp-matmul [n]
 *
 * (n = matrix size; default 1000)
 *
 ****************************************************************************/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Fills n x n square matrix m with random values */
void fill( double* m, int n )
{
    int i, j;

    #pragma omp parallel for collapse(2)
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            m[i*n + j] = (double)rand() / RAND_MAX;
        }
    }
}

int min(int a, int b)
{
    return (a < b ? a : b);
}

/* Cache-efficient computation of r = p * q, where p. q, r are n x n
  matrices. The caller is responsible for allocating the memory for
  r. This function allocates (and the frees) an additional n x n
  temporary matrix. */
void matmul_transpose( double *p, double* q, double *r, int n)
{
    int i, j, k;
    double *qT = (double*)malloc( n * n * sizeof(double) );

    /* transpose q, storing the result in qT */
    #pragma omp parallel for collapse(2)
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }    
        
    /* multiply p and qT row-wise */
    #pragma omp parallel for private(k) collapse(2)
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = 0.0;
            //#pragma omp parallel for
            for (k=0; k<n; k++) {
                r[i*n + j] += p[i*n + k] * qT[j*n + k];
            }
        }
    }

    free(qT);
}

int main( int argc, char *argv[] )
{
    int n = 1000;
    double *p, *q, *r;
    double tstart, tstop;

    if ( argc > 2 ) {
        printf("Usage: %s [n]\n", argv[0]);
        return -1;
    }

    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    p = (double*)malloc( n * n * sizeof(double) );
    q = (double*)malloc( n * n * sizeof(double) );
    r = (double*)malloc( n * n * sizeof(double) );

    fill(p, n);
    fill(q, n);

    tstart = omp_get_wtime();
    matmul_transpose(p, q, r, n);
    tstop = omp_get_wtime();
    printf("Execution time %f\n", tstop - tstart);  

    return 0;
}
