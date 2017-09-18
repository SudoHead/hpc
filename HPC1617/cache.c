/****************************************************************************
 *
 * cache.c - Dense matrix-matrix multiply showing caching effects
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
 * gcc cache.c -o cache
 *
 * Run with:
 * ./cache
 *
 * To see the number of cache misses (currently does not work on disi-hpc):
 * perf -e cache-misses ./cache 
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

/* Fills n x n square matrix m with random values */
void fill( double* m, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            m[i*n + j] = (double)rand() / RAND_MAX;
        }
    }
}

/* compute r = p * q, where p, q, r are n x n matrices. The caller is
   responsible for allocating the memory for r */
void matmul( double *p, double* q, double *r, int n)
{
    int i, j, k;

    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = 0.0;
            for (k=0; k<n; k++) {
                r[i*n + j] += p[i*n + k] * q[k*n + j];
            }
        }
    }
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
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            qT[j*n + i] = q[i*n + j];
        }
    }    

    /* multiply p and qT row-wise */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = 0.0;
            for (k=0; k<n; k++) {
                r[i*n + j] += p[i*n + k] * qT[j*n + k];
            }
        }
    }

    free(qT);
}

int main( void )
{
    const int n = 1000;
    double *p = (double*)malloc( n * n * sizeof(double) );
    double *q = (double*)malloc( n * n * sizeof(double) );
    double *r = (double*)malloc( n * n * sizeof(double) );
    double tstart, tstop;

    fill(p, n);
    fill(q, n);

    tstart = hpc_gettime();
    matmul(p, q, r, n);
    tstop = hpc_gettime();
    printf("Plain matrix multiply: r[0][0] = %f, elapsed time = %f s\n", r[0], tstop - tstart);

    tstart = hpc_gettime();
    matmul_transpose(p, q, r, n);
    tstop = hpc_gettime();
    printf("Cache-efficient matrix multiply: r[0][0] = %f, elapsed time = %f s\n", r[0], tstop - tstart);

    return 0;
}
