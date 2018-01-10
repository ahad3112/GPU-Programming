#ifndef DEVICE_HPP
#define DEVICE_HPP
#include "modelParameters.hpp"
#include "particle.hpp"

/**
* This class is responsible for perform computation in device. Ex: define thread grid, calling kernel
*/
class ComputeDevice {
private:
  ModelParameters* dev_modelParameters;
  float3f* accels;

  // define threads grid size  and block size
  unsigned int sharedMemorySize;
  dim3 blockDim;
  dim3 gridDim;

public:
  Particle* dev_particles;

  ComputeDevice();
  ComputeDevice(cudaGraphicsResource* resource,Particle* h_particles, ModelParameters modelParameters, const int BLOCK_SIZE);
  void compute();
  void releaseMemory();
  ~ComputeDevice();
};

#endif
