/****************************************************************************
 *
 * simd-noauto : examples where automatic vectorization (could) fail
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

#define SIZE 1000000
float vec[SIZE];

typedef float v4f __attribute__((vector_size(16)));
#define VLEN (sizeof(v4f)/sizeof(float))

float f(int n)
{
    int i;
    float x = 0.0;
    float s = 0.0;
    const float delta = 0.1;
    for (i=1; i<n; i++) {
        s += x*x;
        x += delta;
    }
    return s; /* unused */
}

void init(float *v, int n)
{
    int i;
    for (i=0; i<n; i++) {
        v[i] = (float)i / n;
    }
}

int main( void )
{
    init(vec, SIZE);
    printf("%f\n", f(SIZE));
    return 0;
}
