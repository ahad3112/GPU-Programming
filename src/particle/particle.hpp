#ifndef PARTICLE_HPP
#define PARTICLE_HPP
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include "mathUtil.hpp"

enum class ParticleType { IRON, SILICA };

class Particle{
public:
  float3f position;
  float3f velocity;
  ParticleType pType;
  Particle(float3f position, float3f velocity, ParticleType pType);
  ~Particle();
};

#endif
