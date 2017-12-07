
#include <stdio.h>
#include "device.hpp"

__global__ void sampleKernel(){
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Thread: %d\n", gtid);
}

// This method start creating and executing the kernel
void startComputation(){
  printf("Simulation started in device...\n");
  sampleKernel <<< 1, 32 >>> ();
  cudaDeviceSynchronize();

  printf("Simulation finished in device...\n");
}
