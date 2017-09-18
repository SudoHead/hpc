/****************************************************************************
 *
 * mpi-mandelbrot.c - Computation of the Mandelbrot set with MPI
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
 * Compile with
 * mpicc -std=c99 -Wall -Wpedantic mpi-mandelbrot.c -o mpi-mandelbrot
 *
 * run with:
 * mpirun -n 4 ./mpi-mandelbrot
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> /* for uint8_t */

const int maxit = 100;

typedef struct {
    uint8_t r;  /* red   */
    uint8_t g;  /* green */
    uint8_t b;  /* blue  */
} pixel_t;

/**
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first n such that z_n > |bound|, or |maxit| if z_n is below
 * |bound| after |maxit| iterations.
 */
int iterate( float cx, float cy )
{
    float x = 0.0, y = 0.0, xx;
    int it;
    for ( it = 0; (it < maxit) && (x*x + y*y < 2*2); it++ ) {
	xx = x*x - y*y + cx;
	y = 2.0*x*y + cy;
	x = xx;
    }
    return it;
}

/* draw the rows of the Mandelbrot set from ystart (inclusive) to yend
   (excluded), storing the pixels on the bitmap pointed to by pq */
void draw_lines( const int ystart, const int yend, pixel_t* p, const int xsize, const int ysize )
{
    int x, y;
    for ( y = ystart; y < yend; y++) {
        for ( x = 0; x < xsize; x++ ) {
            float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            float cy = 1 - 2.0 * (float)y / (ysize - 1);
            int v = iterate(cx, cy);
            int c = 255-(int)(255.0 * v / maxit);
            p->r = 0;
            p->g = c;
            p->b = c;
            p++;
        }
    }
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz;
    FILE *out = NULL;
    const char* fname="mandelbrot.ppm";
    pixel_t *bitmap = NULL;
    int xsize, ysize;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( argc > 1 ) {
        ysize = atoi(argv[1]);
    } else {
        ysize = 1024;
    }
    xsize = ysize * 1.4;


    if ( 0 == my_rank ) {
        out = fopen(fname, "w");
        if ( !out ) {
            printf("Error: cannot create %s\n", fname);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        bitmap = (pixel_t*)malloc(xsize*ysize*sizeof(*bitmap));

        /* Write the header of the output file */
        fprintf(out, "P6\n");
        fprintf(out, "%d %d\n", xsize, ysize);
        fprintf(out, "255\n");

    }


    int my_ysize = (ysize / comm_sz);
    int my_start = my_rank * my_ysize;
    int my_end = (my_rank + 1) * my_ysize ;

    pixel_t *local_bitmap = (pixel_t*)malloc(xsize * my_ysize * sizeof(*local_bitmap));

    draw_lines(my_start, my_end, local_bitmap, xsize, ysize);

    int local_size = xsize * my_ysize * 3;

    MPI_Gather( local_bitmap,
        local_size,
        MPI_BYTE,
        bitmap,
        local_size,
        MPI_BYTE,
        0,
        MPI_COMM_WORLD
        );


    if ( 0 == my_rank ) {
        fwrite(bitmap, sizeof(*bitmap), xsize*ysize, out);
        fclose(out);
        free(bitmap);
    }

    MPI_Finalize();

    return 0;
}
