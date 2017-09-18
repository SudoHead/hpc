/******************************************************************************
 *
 * omp-coupled-oscillators.c - One-dimensional coupled oscillators system
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
 * This program simulates a population of N coupled oscillators. The
 * system is modeled as N points of equal mass m; point i has position
 * x[][i] and velocity v[][i]. Adjacent masses are connected with a
 * spring whose length at rest is L and whose spring constant is
 * k. The first and last mass are at fixed positions and do not move;
 * the other masses slide without friction along a horizontal line.
 *
 * The system is simulated at discrete time steps. Specifically, the
 * current positions and velocities x[cur][i], v[cur][i] are used to
 * compute the next positions and velocities x[1-cur][i], v[1-cur][i]
 * using nothing more than Newton's second law of motion (F = ma) and
 * Hooke's law for springs (F = kDx, where k is the constant of the
 * spring and Dx the displacement from rest position). The first and
 * last masses are assumed to be fixed, while all other masses can
 * oscillate along the horizontal axis.
 *
 * This program uses Euler's rule to integrate the equations of motion
 * with constant timestep dt.
 *
 * The result is shown in a PPM (portable pixmap) image. The image has
 * N-1 columns; each row shows the energy accumulated in each spring
 * at the corresponding time step (bright colors denote higher
 * energy).
 *
 * This program uses some C99 idioms, so it must be compiled with:
 * gcc -std=c99 -fopenmp -Wall -Wpedantic omp-coupled-oscillators.c -o omp-coupled-oscillators -lm
 *
 * Run with:
 * ./omp-coupled-oscillators
 *
 * The image coupled-oscillators.ppm is created after each run.
 *
 ******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* Number of masses */
#define N 1000
/* Number of initial steps to skip, before starting to take pictures */
#define TRANSIENT 50000 
/* Number of steps to record in the picture */
#define NSTEPS 800
/* Arrays of positions and velocities */
double x[2][N], v[2][N];
/* Index of current array */
int cur = 0;

/* Some physical constants */
const double dt = 0.02; /* Integration time step */
const double k = 0.2;   /* spring constant (large k = stiff spring, small k = soft spring) */
const double m = 1.0;   /* mass */
const double L = 1.0;   /* Length of each spring at rest */

/* Initial conditions: all masses are evenly placed so that the
   springs are at rest; some of the masses are displaced to start the
   movement. */
void init( void )
{
    int i;
    for (i=0; i<N; i++) {
        x[cur][i] = i*L;
        v[cur][i] = 0.0;
    }
    /* displace some of the masses */
    x[cur][N/3  ] -= 0.5*L;
    x[cur][N/2  ] += 0.7*L; 
    x[cur][2*N/3] -= 0.7*L;
}

/* Perform one simulation step: starting from the current positions
   x[] and velocities v[] of the masses, compute the next positions
   xnext[] and velocities vnext[]. the xnext[] and vnext[] arrays must
   be already allocated by the caller. */
void step( double *x, double *v, double *xnext, double *vnext, int n )
{
    /* TO BE COMPLETED */
}

/* Compute x*x */
double squared(double x)
{
    return x*x;
}

int main( void )
{
    int s, i;
    const char* fname = "coupled-oscillators.ppm";
    FILE *fout;

    fout = fopen(fname, "w");
    if (NULL == fout) {
        printf("Cannot open %s for writing\n", fname);
        return -1;
    }

    /* Write the header of the output file */
    fprintf(fout, "P6\n");
    fprintf(fout, "%d %d\n", N-1, NSTEPS);
    fprintf(fout, "255\n");

    /* Initialize the simulation */
    cur = 0;
    init();

    /* Skip the first TRANSIENT steps */
    for (s=0; s<TRANSIENT; s++) {
        const int next = 1 - cur;
        step(x[cur], v[cur], x[next], v[next], N);
        cur = next;
    }

    /* Compute the maximum energy among the springs; this is used to
       normalize the color intensity of the pixels */
    double maxenergy = -1.0;
    for (i=1; i<N; i++) {
        maxenergy = fmax(0.5*k*squared(x[cur][i]-x[cur][i-1]-L), maxenergy);
    }

    /* Write NSTEPS rows in the output image */
    for (s=0; s<NSTEPS; s++) {
        const int next = 1 - cur;
        step(x[cur], v[cur], x[next], v[next], N);
        /* Dump spring energies (light color = high energy) */
        for (i=1; i<N; i++) {
            const double displ = x[cur][i] - x[cur][i-1] - L;
            const double energy = 0.5*k*squared(displ);
            const double v = fmin(energy/maxenergy, 1.0);
            fprintf(fout, "%c%c%c", 0, (int)(255*v*(displ<0)), (int)(255*v*(displ>0)));  
        }
        cur = next;
    }
    fclose(fout);
}
