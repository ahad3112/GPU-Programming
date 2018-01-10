#include <stdio.h>
#include "modelParameters.hpp"

/**
* Implementation of ModelParameters class
*
*/

ModelParameters::ModelParameters(){

}
ModelParameters::ModelParameters(int nParticles, int modelCase) : nParticles(nParticles){
  printf("Setting up model parameters...\n");
  g_mass_si		= (7.4161e+19f * 131072) / nParticles;
  g_mass_fe		= (1.9549e+20f * 131072) / nParticles;

  // Setup modelCase related parameters
  if(modelCase == 1){
    g_radius_collidor = 25576.86545f;          // radius of collidor

    g_k_si		= 2.9114e+14f ;					         // repulsive parameter for silica
    g_k_fe		= 5.8228e+14f;					         // repulsive parameter for iron

    g_sh_depth_si = 0.001f; 					         // shell depth percent of silica
    g_sh_depth_fe = 0.002f;					           // shell depth percent of iron

    g_center_mass_one = float3f(23925.0f,0.0f,9042.70f);
    g_center_mass_two = float3f(-23925.0f,0.0f,-9042.70f);

    g_linear_velocity_one = float3f(-3.24160f,0.0f,0.0f);
    g_linear_velocity_two = float3f(3.24160f,0.0f,0.0f);

    g_angular_velocity_one = float3f(0.0f,8.6036e-6f,0.0f);     //glm::vec3(0.0f,8.6036e-4f,0.0f);
    g_angular_velocity_two = float3f(0.0f,-8.6036e-6f,0.0f);    // glm::vec3(0.0f,-8.6036e-4f,0.0f)

  } else if(modelCase == 2){
    g_radius_collidor = 38747.93036f;
    g_k_si		= 7.2785e+13f ;					         // repulsive parameter for silica
    g_k_fe		= 2.9114e+14f;					         // repulsive parameter for iron

    g_sh_depth_si = 0.001f; 					         // shell depth percent of silica
    g_sh_depth_fe = 0.01f;					           // shell depth percent of iron

    g_center_mass_one = float3f(37678.0f,0.0f,9042.70f);
    g_center_mass_two = float3f(-37678.0f,0.0f,-9042.70f);

    g_linear_velocity_one = float3f(-1.29678f,0.0f,0.0f);
    g_linear_velocity_two = float3f(1.29678f,0.0f,0.0f);

    g_angular_velocity_one = float3f(0.0f,8.6036e-6f,0.0f);     //glm::vec3(0.0f,8.6036e-4f,0.0f);
    g_angular_velocity_two = float3f(0.0f,8.6036e-6f,0.0f);    // glm::vec3(0.0f,8.6036e-4f,0.0f)

  } else if (modelCase == 3){
    g_radius_collidor = 27628.52209f;          // radius of collidor

    g_k_si		= 7.2785e+13f ;					         // repulsive parameter for silica
    g_k_fe		= 2.9114e+14f;					         // repulsive parameter for iron

    g_sh_depth_si = 0.001f; 					         // shell depth percent of silica
    g_sh_depth_fe = 0.01f;					           // shell depth percent of iron

    g_center_mass_one = float3f(24490.7f,9042.70f,9042.70f);
    g_center_mass_two = float3f(-24490.7f,-9042.70f,-9042.70f);

    g_linear_velocity_one = float3f(-1.29664f,0.0f,0.0f);
    g_linear_velocity_two = float3f(1.29664f,0.0f,0.0f);

    g_angular_velocity_one = float3f(0.0f,8.44639e-6f,0.0f);     //glm::vec3(0.0f,8.44639e-4f,0.0f);
    g_angular_velocity_two = float3f(0.0f,8.44639e-6f,0.0f);    // glm::vec3(0.0f,8.44639e-4f,0.0f)

  }

}
