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
 * Inclusive scan with OpenMP.
 *
 * Compile with:
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-inclusive-scan.c -o omp-inclusive-scan
 *
 * Run with:
 * OMP_NUM_THREADS=2 ./omp-inclusive-scan
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Fill v[] with the constant 1 */
void fill(int* v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = 1;
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
    int n = 1000000;
    int i;
    int *v, *s;
    const int n_threads = omp_get_max_threads();
    int blksum[n_threads];
    int blksum_p[n_threads];

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

    /* Parallel scan */
#pragma omp parallel num_threads(n_threads) default(none) shared(v, s, n, blksum, blksum_p) private(i)
    {
        const int thread_id = omp_get_thread_num();
        int local_start = n * thread_id / n_threads;
        int local_end = n * (thread_id + 1) / n_threads;

        /* Each process performs an inclusive scan of its portion of array */
        s[local_start] = v[local_start];
        for (i=local_start+1; i<local_end; i++) {
            s[i] = s[i-1] + v[i];
        }
        blksum[thread_id] = s[local_end-1];
    }

    /* The master performs an exclusive scan */
    blksum_p[0] = 0;
    for (i=1; i<n_threads; i++) {
        blksum_p[i] = blksum_p[i-1] + blksum[i-1];
    }

    /* Each process increments all values of its portion of array */
#pragma omp parallel num_threads(n_threads) default(none) shared(s, n, blksum_p) private(i)
    {
        const int thread_id = omp_get_thread_num();
        int local_start = n * thread_id / n_threads;
        int local_end = n * (thread_id + 1) / n_threads;
        
        for (i=local_start; i<local_end; i++) {
            s[i] += blksum_p[thread_id];
        }
    }
    check(s, n);
    free(v);
    free(s);
    return 0;
}
