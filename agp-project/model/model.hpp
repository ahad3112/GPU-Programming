#ifndef MODEL_HPP
#define MODEL_HPP
#include "particle.hpp"
#include "modelParameters.hpp"

/**
* This class define the Model to be simulated
*
*/

enum class CollidorEnum { ONE, TWO };

class Model {
public:
  // List of model parameters
  ModelParameters modelParameters;
  // List of host particles
  Particle* h_particles;

  Model();
  Model(int nParticles, int modelCase);
  void initModel(int nParticles);
  float3f computeVelocity(float3f position, CollidorEnum value);
  float3f randomShell(float3f rho, float g_radius_core_fe);
  float3f randomSphere(float3f rho, float g_radius_core_fe);
  void releaseMemory();
  ~Model();
};
#endif
