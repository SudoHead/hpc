#include "hpc.h"
#include <stdlib.h>
#include <stdio.h>

__global__ void hello() {
  printf("Hello world\n" );
}

int main() {

  

  hello<<<1,1>>>();
  printf("HERRO MON AMI\n" );
  return 0;
}
