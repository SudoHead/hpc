/*******************************TODO*********************************************
 *
 * cuda-trafficV2.cu - Biham-Middleton-Levine traffic model
 *
 * Written in 2017 by Ma XiangXiang <xiangxiang.ma(at).studio.unibo.it>
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
 * This program uses cuda shared memory.
 *
 * Compile with:
 *
 * 1) make cuda
 * 2) nvcc cuda-trafficV2.cu -o cuda-trafficV2
 *
 * Run with:
 *
 * ./cuda-trafficV2 [nsteps [rho [N]]]
 *
 * where nsteps is the number of simulation steps to execute, rho is
 * the density of vehicles (probability that a cell is occupied by a
 * vehicle), and N is the grid size.
 *
 * NOTE: N must be multiple of BLKSIZE
 *
 ****************************************************************************/
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

#define BLKSIZE 32

typedef unsigned char cell_t;

/* Possible values stored in a grid cell */
enum {
    EMPTY = 0,  /* empty cell            */
    LR,         /* left-to-right vehicle */
    TB          /* top-to-bottom vehicle */
};

/*  Function used by the kernels to map from matrix indexing to array indexing */
__device__ int IDX(int n, int i, int j) {
  int row = (i+n)%n;
  int col = (j+n)%n;
  return row*n + col;
}

/*  Performs an horizontal step: each thread processes one step of a cell.
    It uses shared memory, each block has its own local copy of a portion
    of the domain stored as a matrix.
*/
__global__ void horizontal_stepV2(cell_t *cur, cell_t *next, int n) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  int local_n = n; // saves a local copy to reduce latency

  //Using shared memory to reduce access of global memory.
  //Halo of one element horizontally on each side
  __shared__ cell_t local_cur[BLKSIZE][BLKSIZE+2];

  if(i < local_n && j < local_n) {
      //Initializing the shared memory
      local_cur[ty][tx+1] = cur[IDX(local_n, i, j)];

      //Inizializes the halo on the left and right borders of the block
      if(tx == 0) {
        local_cur[ty][0] = cur[IDX(local_n, i, j-1)];
      } else if(tx == BLKSIZE - 1) {
        local_cur[ty][tx+2] = cur[IDX(local_n, i, j+1)];
      }

      //Waits for all threads to complete the initialization
      __syncthreads();

      //Proceed to excecute the step
      const cell_t left = local_cur[ty][tx];
      const cell_t center = local_cur[ty][tx+1];
      const cell_t right = local_cur[ty][tx+2];

      if(left == LR && center == EMPTY) {
        next[IDX(local_n, i,j)] = LR;
      } else if (center == LR && right == EMPTY ){
        next[IDX(local_n, i,j)] = EMPTY;
      } else {
        next[IDX(local_n, i,j)] = center;
      }
  }
}


/*  Performs an horizontal step: each thread processes one step of a cell.
    It uses shared memory, each block has its own local copy of a portion
    of the domain stored as a matrix.
*/
__global__ void vertical_stepV2(cell_t *cur, cell_t *next, int n) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = threadIdx.y;
  const int tx = threadIdx.x;
  int local_n=n; // saves a local copy to reduce latency

  //Using shared memory to reduce access of global memory.
  //Halo of one element vertically on each side
  __shared__ cell_t local_cur[BLKSIZE+2][BLKSIZE];

  if(i < local_n && j <  local_n) {
      //Initializing the shared memory
      local_cur[ty+1][tx] = cur[IDX(local_n, i, j)];

      //Inizializes the halo on the top and bottom borders of the block
      if(ty == 0) {
        local_cur[0][tx] = cur[IDX(local_n, i-1, j)];
      } else if(ty == BLKSIZE - 1) {
        local_cur[ty+2][tx] = cur[IDX(local_n, i+1, j)];
      }

      //Waits for all threads to complete the initialization
      __syncthreads();

      //Proceed to excecute the step
      const cell_t up = local_cur[ty][tx];
      const cell_t center = local_cur[ty+1][tx];
      const cell_t down = local_cur[ty+2][tx];

      if(up == TB && center == EMPTY) {
        next[IDX(local_n, i,j)] = TB;
      } else if (center == TB && down == EMPTY ){
        next[IDX(local_n, i,j)] = EMPTY;
      } else {
        next[IDX(local_n, i,j)] = center;
      }
    }
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
          grid[i*n + j] = getRand() <= 0.5 ? TB : LR;
        } else {
          grid[i*n + j] = EMPTY;
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
            switch( grid[i*n + j] ) {
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
                printf("Error: unknown cell state %u\n", grid[i*n + j]);
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
    cell_t *d_cur, *d_next;
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
        if(N % BLKSIZE != 0) {
          fprintf(stderr, "N must be multiple of BLOCKSIZE (%d)\n", BLKSIZE);
          return 0;
        }
    }

    /*  Sets a 2D square block of BLKSIZE dimension */
    dim3 threadPerBlock(BLKSIZE, BLKSIZE);
    /*  Sets a 2D cuda grid big enough to associate each cell to a thread  */
    dim3 grid((N+BLKSIZE-1)/BLKSIZE, (N+BLKSIZE-1)/BLKSIZE);
    const size_t size = N*N*sizeof(cell_t);

    /* Allocate space for device copies of d_cur, d_next */
    cudaMalloc((void **)&d_cur, size);
    cudaMalloc((void **)&d_next, size);

    /* Allocate grids */
    cur = (cell_t*)malloc(size);
    next = (cell_t*)malloc(size);

    setup(cur, N, rho);

    /*  Copies the host's cur and next to the device  */
    cudaMemcpy(d_cur, cur, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_next, next, size, cudaMemcpyHostToDevice);

    tstart = hpc_gettime();
    for(int k=0; k < nsteps; k++) {
      //Async kernel call
      horizontal_stepV2<<<grid, threadPerBlock>>>(d_cur, d_next, N);
      //Waits for the kernel to complete
      cudaDeviceSynchronize();

      //Async kernel call
      vertical_stepV2<<<grid, threadPerBlock>>>(d_next, d_cur, N);
      //Waits for the kernel to complete
      cudaDeviceSynchronize();
    }

    tend = hpc_gettime();
    fprintf(stdout, "Execution time (s): %f\n", tend - tstart);

    /*  Copies the result back to the Host from the device  */
    cudaMemcpy(cur, d_cur, size, cudaMemcpyDeviceToHost);

    /* dump last state */
    s = nsteps;
    snprintf(buf, BUFLEN, "cuda-trafficV2-%d.ppm", s);
    dump(cur, N, buf);

    /* Free memory */
    free(cur);
    free(next);
    cudaFree(d_cur);
    cudaFree(d_next);
    return 0;
}
