/****************************************************************************
 *
 * mpi-type-contiguous.c - Simple demo of the MPI_Type_contiguous function
 *
 * Based on https://computing.llnl.gov/tutorials/mpi/#Derived_Data_Types
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-type-contiguous.c -o mpi-type-contiguous
 *
 * Run with:
 * mpirun -n 4 ./mpi-type-contiguous
 *
 ****************************************************************************/

#include <stdio.h>
#include <mpi.h>

#define SIZE 4

int main( int argc, char *argv[] )  
{
    int numtasks, rank, source=0, tag=1, i;

    float a[SIZE][SIZE] =
        {{ 1.0,  2.0,  3.0,  4.0},
         { 5.0,  6.0,  7.0,  8.0},
         { 9.0, 10.0, 11.0, 12.0},
         {13.0, 14.0, 15.0, 16.0}};
    float b[SIZE];
    
    MPI_Status stat;
    MPI_Datatype rowtype;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    
    /* create contiguous derived data type */
    MPI_Type_contiguous(SIZE,      /* count   */
			MPI_FLOAT, /* oldtype */
			&rowtype   /* newtype */
			);
    MPI_Type_commit(&rowtype);

    if ( rank == 0 && numtasks == 1 ) {
        printf("You must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        /* The master sends one element of rowtype to all other tasks */
        for (i=1; i<numtasks; i++)
            MPI_Send(&a[i % SIZE][0], 1, rowtype, i, tag, MPI_COMM_WORLD);
    } else {    
        /* all other tasks receive rowtype data from process 0 */
        MPI_Recv(b, SIZE, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &stat);
        printf("rank= %d received %3.1f %3.1f %3.1f %3.1f\n",
               rank, b[0], b[1], b[2], b[3]);
    }
    
    /* free datatype when done using it */
    MPI_Type_free(&rowtype);
    MPI_Finalize();
    return 0;
}
