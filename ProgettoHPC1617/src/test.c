#include <stdlib.h>
#include <stdio.h>

float getRand() {
  return ((float) rand() / (RAND_MAX));
}


int main() {
  for(int i=0; i < 100; i++) {
    float r = getRand();
    printf("r = %f\n", r);
  }
}
