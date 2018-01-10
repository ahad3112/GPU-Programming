#ifndef MODEL_PARAMETERS_HPP
#define MODEL_PARAMETERS_HPP

#include "float3f.hpp"

/**
* THis class hold the all related parameters related to a particular model case for simulation
*
*/

class ModelParameters{
public:
  int nParticles;
  // list of model related parameters
  float G = 6.674e-20f; 							             // Universal Gravitational constant: unit [m3⋅kg−1⋅s−2.]

  float g_diameter 	= 800.7800f;				           // diameter of an element ???? This is not scaled ??????
  float g_diameter2 	= g_diameter * g_diameter;	 // diameter square of an element

  float g_reduce_k_si	= 0.01f;					           // persent of reduction of the silicate repulsive force
  float g_reduce_k_fe	= 0.02f;					           // persent of reduction of the iron repulsive force

  float g_epsilon		= 47.0975f;					           // epsilon to avoid singularity
  float g_epsilon2		= g_epsilon * g_epsilon;		 // square of epsilon to avoid singularity
  float g_time_step	= 10.8117f;

  float PERCENT_IRON = 0.30f;

  // mass depends on nParticles
  float g_mass_si;
  float g_mass_fe;

  // The following values depends on modelCase
  float g_radius_collidor;          // radius of collidor

  float g_k_si;					                          // repulsive parameter for silica
  float g_k_fe;					                          // repulsive parameter for iron

  float g_sh_depth_si; 					                  // shell depth percent of silica
  float g_sh_depth_fe;					                  // shell depth percent of iron

  float3f g_center_mass_one;                      // center of mass of impactor 1
  float3f g_center_mass_two;                      // center of mass of impactor 2

  float3f g_linear_velocity_one;                  // linear velocity  of impactor 1
  float3f g_linear_velocity_two;                  // linear velocity  of impactor 2

  float3f g_angular_velocity_one;                 // angular velocity  of impactor 1
  float3f g_angular_velocity_two;                 // angular velocity  of impactor 2

  // Constructor
  ModelParameters();
  ModelParameters(int nParticles, int modelCase);
};

#endif
