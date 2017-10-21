/****************************************************************************
 *
 * traffic.c - Biham-Middleton-Levine traffic model
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
 * ---------------------------------------------------------------------------
 *
 * This program implements the Biham-Middleton-Levine traffic model
 * The BML traffic model is a simple three-state 2D cellular automaton
 * over a toroidal square lattice space. Initially, each cell is
 * either empty, or contains a left-to-right (LR) or top-to-bottom
 * (TB) moving vehicle. The model evolves at discrete time steps. Each
 * step is logically divided into two phases: in the first phase only
 * LR vehicles move, provided that the destination cell is empty; in
 * the second phase, only TB vehicles move, again provided that the
 * destination cell is empty.
 *
 * This program is not complete: some functions are missing and must
 * be implemented.
 *
 * Compile with:
 *
 * gcc -fopenmp -std=c99 -Wall -Wpedantic traffic.c -o traffic
 *
 * Run with:
 *
 * ./traffic [nsteps [rho [N]]]
 *
 * where nsteps is the number of simulation steps to execute, rho is
 * the density of vehicles (probability that a cell is occupied by a
 * vehicle), and N is the grid size.
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

typedef unsigned char cell_t;

/* Possible values stored in a grid cell */
enum {
    EMPTY = 0,  /* empty cell            */
    LR,         /* left-to-right vehicle */
    TB          /* top-to-bottom vehicle */
};

/*Return the right cyclic matrix index*/
unsigned int IDX(int n, int i, int j) {
  int row = (i+n)%n;
  int col = (j+n)%n;
  return row*n + col;
}

/* Move all left-to-right vehicles that are not blocked */
void horizontal_step( cell_t *cur, cell_t *next, int n )
{
    #pragma omp parallel for collapse(2)
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        if(cur[IDX(n, i,j-1)] == LR && cur[IDX(n, i,j)] == EMPTY) {
          next[IDX(n, i,j)] = LR;
        } else if (cur[IDX(n, i,j)] == LR && cur[IDX(n, i,j+1)] == EMPTY ){
          next[IDX(n, i,j)] = EMPTY;
        } else {
          next[IDX(n, i,j)] = cur[IDX(n, i,j)];
        }
      }
    } // end 1st for
}

/* Move all top-to-bottom vehicles that are not blocked */
void vertical_step( cell_t *cur, cell_t *next, int n )
{
    #pragma omp parallel for collapse(2)
    for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
        if(cur[IDX(n, i-1,j)] == TB && cur[IDX(n, i,j)] == EMPTY) {
          next[IDX(n, i,j)] = TB;
        } else if (cur[IDX(n, i,j)] == TB && cur[IDX(n, i+1,j)] == EMPTY ){
          next[IDX(n, i,j)] = EMPTY;
        } else {
          next[IDX(n, i,j)] = cur[IDX(n, i,j)];
        }
      }
    } // end 1st for
}

/*Returns a random number between 0 and 1*/
float getRand() {
  return ((float) rand() / (RAND_MAX));
}

/* Initialize |grid| with vehicles with density |rho|. |rho| must be
   in the range [0, 1] (rho = 0 means no vehicle, rho = 1 means that
   every cell is occupied by a vehicle). The direction is chosen with
   equal probability. */
void setup( cell_t* grid, int n, float rho )
{
    /* TODO */
    for(int i=0; i<n;i++) {
      for(int j=0; j<n; j++) {

        if(getRand() <= rho) {
          grid[IDX(n, i,j)] = getRand() <= 0.5 ? TB : LR;
        } else {
          grid[IDX(n, i,j)] = EMPTY;
        }
      }
    }

}

/* Dump |grid| as a PPM (Portable PixMap) image written to file
   |filename|. LR vehicles are shown as red pixels, while TB vehicles
   are shown in blue. Empty cells are white. */
void dump( const cell_t *grid, int n, const char* filename )
{
    int i, j;
    FILE *out = fopen( filename, "w" );
    if ( NULL == out ) {
        printf("Cannot create \"%s\"\n", filename);
        abort();
    }
    fprintf(out, "P6\n");
    fprintf(out, "%d %d\n", n, n);
    fprintf(out, "255\n");
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            switch( grid[IDX(n, i,j)] ) {
            case EMPTY:
                fprintf(out, "%c%c%c", 255, 255, 255);
                break;
            case TB:
                fprintf(out, "%c%c%c", 0, 0, 255);
                break;
            case LR:
                fprintf(out, "%c%c%c", 255, 0, 0);
                break;
            default:
                printf("Error: unknown cell state %u\n", grid[IDX(n, i,j)]);
                abort();
            }
        }
    }
    fclose(out);
}

#define BUFLEN 256

int main( int argc, char* argv[] )
{
    cell_t *cur, *next;
    char buf[BUFLEN];
    int s, N = 256, nsteps = 512;
    float rho = 0.2;
    double tstart, tend;

    if ( argc > 4 ) {
        printf("Usage: %s [nsteps [rho [N]]]\n", argv[0]);
        return -1;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        rho = atof(argv[2]);
    }

    if ( argc > 3 ) {
        N = atoi(argv[3]);
    }

    const size_t size = N*N*sizeof(cell_t);

    /* Allocate grids */
    cur = (cell_t*)malloc(size);
    next = (cell_t*)malloc(size);

    setup(cur, N, rho);
    tstart = hpc_gettime();

    for (s=0; s<nsteps; s++) {
      horizontal_step(cur, next, N);
      vertical_step(next, cur, N);
    }

    tend = hpc_gettime();
    fprintf(stdout, "Execution time (s): %f\n", tend - tstart);
    /* dump last state */
    snprintf(buf, BUFLEN, "omp-traffic-%d.ppm", s);
    dump(cur, N, buf);

    /* Free memory */
    free(cur);
    free(next);
    return 0;
}
