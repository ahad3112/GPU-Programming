#ifndef CPUSIM_HPP
#define CPUSIM_HPP

#include "particle.hpp"
#include "modelParameters.hpp"

/*
* The helper method is for simulation in CPU only to compare performance result with cuda
*/
void updateEarthMoonSystemCPU(Particle* h_particles, ModelParameters* modelParameters);

#endif
