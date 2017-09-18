/****************************************************************************
 *
 * simd-vsum-vector.c - Vector sum using vector data type
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 *
 * gcc -std=c99 -Wall -Wpedantic -mtune=native simd-vsum-vector.c -o simd-vsum-vector
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1 << 16)

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

float vsum(const float *v, int n)
{
    v4f vs = {0.0, 0.0, 0.0, 0.0};
    float ss = 0.0;
    const v4f *vv = (v4f*)v;
    int i;
    for (i=0; i<n-VLEN-1; i += VLEN) {
        vs += *vv;
        vv++;
    }
    for ( ; i<n; i++) {
        vs[0] += v[i];
    }
    /* Loop unroll optimization will hopefully take care of this; we
       will see in the future how to implement horizontal sums more
       efficiently (but less portably). */
    for (i=0; i<VLEN; i++) {
        ss += vs[i];
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
