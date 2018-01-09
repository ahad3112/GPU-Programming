#ifndef DEVICE_HPP
#define DEVICE_HPP
#include "modelParameters.hpp"
#include "particle.hpp"

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
  ComputeDevice(cudaGraphicsResource* resource,Particle* h_particles, ModelParameters modelParameters);
  void compute();
  void releaseMemory();
  ~ComputeDevice();
};

#endif
