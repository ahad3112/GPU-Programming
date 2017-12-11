#ifndef DEVICE_HPP
#define DEVICE_HPP


class float3f{
public:
  float x;
  float y;
  float z;
  __host__ __device__ float3f(float x, float y, float z);

  __host__ __device__ float3f();

  __host__ __device__ ~float3f();
};

enum class ParticleType { IRON, SILICA };

class Particle{
public:
  float3f position;
  float3f velocity;
  ParticleType pType;
  __host__ __device__ Particle(float3f position, float3f velocity, ParticleType pType);

  __host__ __device__ ~Particle();
};



Particle* allocateHostMemory(int N);
void releaseHostMemory(Particle* h_particles);
void startComputation(Particle* h_particles, int N);

#endif
