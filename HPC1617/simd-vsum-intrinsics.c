/****************************************************************************
 *
 * simd-vsum-intrinsics.c - Vector sum using SSE intrinsics
 *
 * Written in 2016 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#define SIZE (1 << 16)

/* Return the sum of the n values stored in v,m using SIMD intrinsics.
   no particular alignment is required for v; n can be arbitrary */
float vsum(float *v, int n)
{
    __m128 vv, vs;
    float ss = 0.0;
    int i;

    vs = _mm_setzero_ps();
    for (i=0; i<n-4+1; i += 4) {
        vv = _mm_loadu_ps( &v[i] );     /* load four floats into vv */
        vs = _mm_add_ps(vv, vs);        /* vs = vs + vv */
    }
    ss = vs[0] + vs[1] + vs[2] + vs[3];
    for ( ; i<n; i++) {
        ss += v[i];
    }
    return ss;
}

float vsum_scalar(const float *v, int n)
{
    float s = 0.0;
    int i;
    for (i=0; i<n; i++) {
        s += v[i];
    }
    return s;
}

void fill(float *v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = i%10;
    }
}

int main( void )
{
    float *vec = malloc(SIZE*sizeof(*vec));
    fill(vec, SIZE);
    printf("vector sum = %f\n", vsum(vec, SIZE));
    printf("scalar sum = %f\n", vsum_scalar(vec, SIZE));
    free(vec);
    return 0;
}
