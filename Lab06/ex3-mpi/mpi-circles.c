/****************************************************************************
 *
 * mpi-circles.c - Compute the area of the union of a set of circles
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-circles.c -o mpi-circles
 *
 * Run with:
 * mpirun -n 4 ./mpi-circles 10000 circles-1000.in
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* for time() */

float sq(float x)
{
    return x*x;
}

int inside( float* x, float* y, float *r, int n, int k )
{
    int i, np, c=0;
    float px, py;
    for (np=0; np<k; np++) {
        px = 100.0*rand()/(float)RAND_MAX;
        py = 100.0*rand()/(float)RAND_MAX;
        for (i=0; i<n; i++) {
            if ( sq(px-x[i]) + sq(py-y[i]) <= sq(r[i]) ) {
                c++;
                break;
            }
        }
    }
    return c;    
}

int main( int argc, char* argv[] )
{
    float *x, *y, *r;
    int N, K, c;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* Initialize the RNG */
    srand(time(NULL));

    if ( (0 == my_rank) && (argc != 3) ) {
        printf("Usage: %s [npoints] [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    K = atoi(argv[1]);

    /* This is not a true parallel version: the master does everything */
    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[2], "r");
        int i;
        if ( !in ) {
            printf("Cannot open %s for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        fscanf(in, "%d", &N);
        x = (float*)malloc(N * sizeof(*x));
        y = (float*)malloc(N * sizeof(*y));
        r = (float*)malloc(N * sizeof(*r));
        for (i=0; i<N; i++) {
            fscanf(in, "%f %f %f", &x[i], &y[i], &r[i]);
        }
        fclose(in);

        c = inside(x, y, r, N, K);

        /* print the area */
        printf("%d points, %d inside, area %f\n", K, c, 1.0e6*c/K);
    }

    MPI_Finalize();

    return 0;
}
