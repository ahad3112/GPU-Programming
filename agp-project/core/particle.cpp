
#include "particle.hpp"
  /**
  * Implementation of Particle class
  */
  __host__ __device__ Particle::Particle(float3f position,float3f color, float3f velocity, ParticleType pType) :
  position(position), color(color), velocity(velocity), pType(pType)
  {
    //std::cout<<"Initializing particle..."<<std::endl;
  }

  __host__ __device__ Particle::Particle(){
    position = float3f();
    color = float3f();
    velocity = float3f();
    pType = ParticleType::IRON;
  }

  __host__ __device__ Particle::~Particle(){

  }
