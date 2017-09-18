/****************************************************************************
 *
 * simd-hsum.c - horizontal sum with SSE2
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

void print_xmm(__m128i v)
{
    int* vv = (int*)&v;
    printf("v0=%d v1=%d v2=%d v3=%d\n", vv[0], vv[1], vv[2], vv[3]);
}

int main( void )
{
    __m128i v, vp;
    int r;

    v  = _mm_set_epi32(19,-1, 77, 34); /* [34|77|-1|19] */
    print_xmm(v);
    vp = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));   /*  11.11.01.01  */
    print_xmm(vp);
    v = _mm_add_epi32(v, vp);
    print_xmm(v);
    vp = _mm_shuffle_epi32(v, 0xaa);   /*  10.10.10.10  */
    print_xmm(vp);
    v = _mm_add_epi32(v, vp);
    print_xmm(v);
    r = _mm_cvtsi128_si32(v);      /* get v0        */
    printf("%d\n", r);
    return 0;
}
