/****************************************************************************
 *
 * simd-vsum-auto : Vector sum; this program is used to test compiler
 * auto-vectorization.
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
 * gcc -march=native -O2 -ftree-vectorize -ftree-vectorizer-verbose=1 simd-vsum-auto.c -o simd-vsum-auto
 *
 * and observe that the loop in vec_sum is not vectorized. To see why,
 * compile with:
 *
 * gcc -march=native -O2 -ftree-vectorize -ftree-vectorizer-verbose=2 simd-vsum-auto.c -o simd-vsum-auto
 *
 * To forse vectorization anyway, use -funsafe-math-optimizations
 *
 * gcc -funsafe-math-optimizations -march=native -O2 -ftree-vectorize -ftree-vectorizer-verbose=2 simd-vsum-auto.c -o simd-vsum-auto
 *
 * To see the assembly output:
 *
 * gcc -funsafe-math-optimizations -march=native -c -Wa,-adhln -g -O2 -ftree-vectorize -ftree-vectorizer-verbose=2 simd-vsum-auto.c > simd-vsum-auto.s
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#define SIZE (1 << 16)

float vsum(float *v, int n)
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
    printf("sum = %f\n", vsum(vec, SIZE));
    free(vec);
    return 0;
}
