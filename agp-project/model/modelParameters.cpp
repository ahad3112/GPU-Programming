#include <stdio.h>
#include "modelParameters.hpp"

ModelParameters::ModelParameters(){

}
ModelParameters::ModelParameters(int nParticles, int modelCase) : nParticles(nParticles){
  printf("Setting up model parameters...\n");
  g_mass_si		= (7.4161e+19f * 131072) / nParticles;
  g_mass_fe		= (1.9549e+20f * 131072) / nParticles;

  // Setup modelCase related parameters
  if(modelCase == 1){
    g_center_mass_one = float3f(23925.0f,0.0f,9042.70f);
    g_center_mass_two = float3f(-23925.0f,0.0f,-9042.70f);

    g_linear_velocity_one = float3f(-3.24160f,0.0f,0.0f);
    g_linear_velocity_two = float3f(3.24160f,0.0f,0.0f);

    g_angular_velocity_one = float3f(0.0f,8.6036e-6f,0.0f);     //glm::vec3(0.0f,8.6036e-4f,0.0f);
    g_angular_velocity_two = float3f(0.0f,-8.6036e-6f,0.0f);    // glm::vec3(0.0f,-8.6036e-4f,0.0f)
  }

}
