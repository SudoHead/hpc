/****************************************************************************
 *
 * mpi-bbox.c - Compute the bounding box of a set of rectangles
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-bbox.c -o mpi-bbox -lm
 *
 * Run with:
 * mpirun -n 4 ./mpi-bbox bbox-1000.in
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fminf() */
#include <mpi.h>

/* Compute the bounding box of n rectangles whose opposite vertices
   have coordinates (x1, y1), (x2, y2). Put the results in the output
   parameters (xb1, yb1), (xb2, yb2) */   
void bbox( float *x1, float *y1, float* x2, float *y2, int n,
           float *xb1, float *yb1, float *xb2, float *yb2 )
{
    int i;
    *xb1 = x1[0];
    *yb1 = y1[0];
    *xb2 = x2[0];
    *yb2 = y2[0];
    for (i=1; i<n; i++) {
        *xb1 = fminf( *xb1, x1[i] );
        *yb1 = fminf( *yb1, y1[i] );
        *xb2 = fmaxf( *xb2, x2[i] );
        *yb2 = fmaxf( *yb2, y2[i] );
    }
}

int main( int argc, char* argv[] )
{
    float *x1, *y1, *x2, *y2;
    float xb1, yb1, xb2, yb2;
    int N;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( (0 == my_rank) && (argc != 2) ) {
        printf("Usage: %s [inputfile]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    x1 = y1 = x2 = y2 = NULL;

    float *local_x1, *local_y1, *local_x2, *local_y2;
    float local_n;


    /* This is not a true parallel version: the master does everything! */
    if ( 0 == my_rank ) {
        FILE *in = fopen(argv[1], "r");
        int i;
        if ( !in ) {
            printf("Cannot open %s for reading\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        fscanf(in, "%d", &N);
        x1 = (float*)malloc(N * sizeof(*x1));
        y1 = (float*)malloc(N * sizeof(*y1));
        x2 = (float*)malloc(N * sizeof(*x2));
        y2 = (float*)malloc(N * sizeof(*y2));
        for (i=0; i<N; i++) {
            fscanf(in, "%f %f %f %f", &x1[i], &y1[i], &x2[i], &y2[i]);
        }
        fclose(in);
        

        /* Compute bounding box */
        //bbox( x1, y1, x2, y2, N, &xb1, &yb1, &xb2, &yb2 );

        /* Print bounding box */
        //printf("bbox: %f %f %f %f\n", xb1, yb1, xb2, yb2);
    }

    /*Broadcast*/
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

 	local_n = N / comm_sz;

    local_x1 = (float*)malloc(local_n * sizeof(float));
    local_y1 = (float*)malloc(local_n * sizeof(float));
    local_x2 = (float*)malloc(local_n * sizeof(float));
    local_y2 = (float*)malloc(local_n * sizeof(float));

    float local_xb1, local_yb1, local_xb2, local_yb2;

        /*Scatter vector x1*/
        MPI_Scatter(x1,
        	local_n,
        	MPI_FLOAT,
        	local_x1,
        	local_n,
        	MPI_FLOAT,
        	0,
        	MPI_COMM_WORLD
        	);

        /*Scatter vector y1*/
        MPI_Scatter(y1,
        	local_n,
        	MPI_FLOAT,
        	local_y1,
        	local_n,
        	MPI_FLOAT,
        	0,
        	MPI_COMM_WORLD
        	);

        /*Scatter vector x2*/
        MPI_Scatter(x2,
        	local_n,
        	MPI_FLOAT,
        	local_x2,
        	local_n,
        	MPI_FLOAT,
        	0,
        	MPI_COMM_WORLD
        	);

        /*Scatter vector y2*/
        MPI_Scatter(y2,
        	local_n,
        	MPI_FLOAT,
        	local_y2,
        	local_n,
        	MPI_FLOAT,
        	0,
        	MPI_COMM_WORLD
        	);


    /* Compute bounding box */
    bbox( local_x1, local_y1, local_x2, local_y2, local_n, &local_xb1, &local_yb1, &local_xb2, &local_yb2 );

   	MPI_Reduce(&local_xb1,
   		&xb1,
   		1,
   		MPI_FLOAT,
   		MPI_MIN,
   		0,
   		MPI_COMM_WORLD
   		);

   	MPI_Reduce(&local_yb1,
   		&yb1,
   		1,
   		MPI_FLOAT,
   		MPI_MIN,
   		0,
   		MPI_COMM_WORLD
   		);

   	MPI_Reduce(&local_xb2,
   		&xb2,
   		1,
   		MPI_FLOAT,
   		MPI_MAX,
   		0,
   		MPI_COMM_WORLD
   		);

   	MPI_Reduce(&local_yb2,
   		&yb2,
   		1,
   		MPI_FLOAT,
   		MPI_MAX,
   		0,
   		MPI_COMM_WORLD
   		);


 	if(my_rank == 0) {
 		printf("bbox: %f %f %f %f\n", xb1, yb1, xb2, yb2);	
 	}

    MPI_Finalize();

    return 0;
}
