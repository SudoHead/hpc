/****************************************************************************
 *
 * cuda-hello1.cu - Hello world with CUDA (with dummy device code)
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
 * Based on the example shown in the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Compile with:
 * nvcc cuda-hello1.cu -o cuda-hello1
 *
 * Run with:
 * ./cuda-hello1
 *
 ****************************************************************************/

#include <stdio.h>

__global__ void mykernel( void ) { }

int main( void )
{
    mykernel<<<1,1>>>( );
    printf("Hello, world!\n");
    return 0;
}
