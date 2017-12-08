#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "particle.hpp"

Particle* allocateHostMemory(int N);
void releaseHostMemory(Particle* h_particles);
void startComputation(Particle* h_particles, int N);

#endif
