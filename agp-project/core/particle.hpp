#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include "float3f.hpp"

  enum class ParticleType { IRON, SILICA };
  class Particle{
    public:
    float3f position;
    float3f color;
    float3f velocity;
    ParticleType pType;
    __host__ __device__ Particle();
    __host__ __device__ Particle(float3f position, float3f color,float3f velocity, ParticleType pType);

    __host__ __device__ ~Particle();
  };
#endif
