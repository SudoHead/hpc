/****************************************************************************
 *
 * mpi-cart.c - Demo for cartesian coordinates
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
 * This example has been adapted from:
 * https://computing.llnl.gov/tutorials/mpi/#Virtual_Topologies
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-cart.c -o mpi-cart
 *
 * Run with:
 * mpirun -n 16 ./mpi-cart
 *
 ****************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

enum {
    NORTH = 0,
    EAST,
    SOUTH,
    WEST
};

int main( int argc, char* argv[] )
{
    const int SIZE = 16;
    int comm_sz;
    int dims[2] = {4, 4};
    int periods[2] = {0, 1}; /* first dimension does not wrap around */
    MPI_Comm cart_comm;
    int my_cart_rank;
    int my_cart_coords[2];
    int nbors[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( SIZE != comm_sz ) {
        printf("You must specify %d processes. Aborting.\n", SIZE);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Create a cartesian grid with wrap-around */
    MPI_Cart_create(MPI_COMM_WORLD, /* original communicator */
                    2,          /* how many dimensions */
                    dims,       /* array of dimensions */
                    periods,    /* wrap around */
                    0,          /* no reordering */
                    &cart_comm  /* new communicator */
                    );

    /* get my rank on the cart_comm communicator; we asked for
       reordering, therefore it is important to use the cart_comm and
       not MPI_COMM_WORLD communicator. */
    MPI_Comm_rank(cart_comm, &my_cart_rank);

    /* get my coords in the 2D grid */
    MPI_Cart_coords(cart_comm, my_cart_rank, 2, my_cart_coords);

    /* get rank of the neighbors */
    MPI_Cart_shift(cart_comm, 0, 1, &nbors[NORTH], &nbors[SOUTH]); /* direction 0 = vertical */
    MPI_Cart_shift(cart_comm, 1, 1, &nbors[WEST], &nbors[EAST]); /* direction 1 = horizontal */
    
    printf("rank=%2d  coords=(%d, %d)  neighbors(N,E,S,W)=%2d, %2d, %2d, %2d\n",
           my_cart_rank, 
           my_cart_coords[0], my_cart_coords[1], 
           nbors[NORTH], nbors[EAST], nbors[SOUTH], nbors[WEST]);

    MPI_Finalize();

    return 0;
}
