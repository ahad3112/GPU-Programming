#include <iostream>
#include "particle.hpp"

Particle::Particle(float3f position, float3f velocity, ParticleType pType) :
position(position), velocity(velocity), pType(pType)
{
  //std::cout<<"Initializing particle..."<<std::endl;
}

Particle::~Particle(){

}
