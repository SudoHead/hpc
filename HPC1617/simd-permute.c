/****************************************************************************
 *
 * simd-permute.c - Demo of SSE permute operation
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
#include <x86intrin.h>

void print_simd( __m128i v )
{
    const int* vv = (int*)&v;
    printf("%d %d %d %d\n", vv[0], vv[1], vv[2], vv[3]);
}

int main( void )
{
    __m128i v, v1, v2, v3, v4;

    v = _mm_set_epi32(19, -1, 77, 34);
    print_simd(v);
    v1 = _mm_shuffle_epi32(v, 0xa0); /* 10.10.00.00 */
    print_simd(v1);
    v2 = _mm_add_epi32(v, v1);
    print_simd(v2);
    v3 = _mm_shuffle_epi32(v2, 0x55); /* 01.01.01.01 */
    print_simd(v3);
    v4 = _mm_add_epi32(v2, v3);
    print_simd(v4);
    return 0;
}
