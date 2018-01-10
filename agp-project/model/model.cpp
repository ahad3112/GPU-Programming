#include <iostream>
#include <stdio.h>
#include <string>
#include <random>
#include "model.hpp"
/**
* Implementation of model class
*
*/

// Variables related to particle generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);
const float PI = 3.1415927;

Model::Model() : Model(960, 1){

}

Model::Model(int nParticles, int modelCase){

  char*  currentCase;
  if(modelCase == 1){
    currentCase = "Plannar single-tailed collision";
  } else if (modelCase == 2){
    currentCase = "Plannar double-tailed collision";
  } else if(modelCase == 3){
    currentCase = "Parallel off-planar collision";
  } else{
    printf("Wrong Model case. Available case values are 1,2 and 3");
  }

  // set up model parameters
  modelParameters = ModelParameters(nParticles, modelCase);

  // Initialize particles list
  initModel(nParticles);

  printf("Initialized %s with nParticles = %d.\n",currentCase, modelParameters.nParticles);
}


// This method return random float between the two given float
float getRND(float a, float b) {
  float random = ((float) rand()) / (float) RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

float rndFloat(){
  float randomval=dis(gen);
	return randomval;
  //std::cout<<dis(gen)<<"\n";
}


float3f rndFloat3(){
	return float3f(rndFloat(),rndFloat(),rndFloat());
}


/**
 * Uses a random spherical distribution to create random locations in a sphere
 * returns : a random position in the sphere
 * */
float3f Model::randomSphere(float3f rho, float g_radius_core_fe){

  float3f res;
  float pre,suf;
  float mu    = (1 - 2 * rho.y);
  float angle = 2*PI*rho.z;


  pre     = g_radius_core_fe * std::cbrt(rho.x);
  suf     = std::sqrt(1-std::pow(mu,2));
  res.x   = pre * suf * std::cos(angle);
  res.y   = pre * suf * std::sin(angle);
  res.z   = pre * mu;

  return res;
}

/**
 * Uses a random spherical distribution to create random locations in a shell of
 * inner radius : g_radius_core_fe
 * outer radius : g_radius_collidor
 * returns : a random position in the shell
 * */
float3f Model::randomShell(float3f rho, float g_radius_core_fe){

  float3f res;
  float pre,suf;

  float mu    = (1 - 2 * rho.y);
  float angle = 2*PI*rho.z;

  pre     = std::cbrt(std::pow(g_radius_core_fe,3) + (std::pow(modelParameters.g_radius_collidor,3) - std::pow(g_radius_core_fe,3)) * rho.x);
  suf     = std::sqrt(1 - std::pow(mu,2));
  res.x   = pre * suf * std::cos(angle);
  res.y   = pre * suf * std::sin(angle);
  res.z   = pre * mu;

  return res;

}

float3f Model::computeVelocity(float3f position, CollidorEnum value){

  float3f l_center_mass;
	float3f l_linear_velocity;
	float3f l_angular_velocity;
	float3f velocity;
	float r_xz;
	float theta;
	if(value == CollidorEnum::ONE){
		 l_center_mass = modelParameters.g_center_mass_one;
		 l_linear_velocity = modelParameters.g_linear_velocity_one;
		 l_angular_velocity = modelParameters.g_angular_velocity_one;
	}else{
		 l_center_mass = modelParameters.g_center_mass_two;
		 l_linear_velocity = modelParameters.g_linear_velocity_two;
		 l_angular_velocity = modelParameters.g_angular_velocity_two;
	}

	velocity = l_linear_velocity;

	r_xz = std::sqrt(pow((position.x - l_center_mass.x),2) + pow((position.z - l_center_mass.z),2)); // why not y ??
	theta = std::atan2((position.z - l_center_mass.z),(position.x - l_center_mass.x));
	velocity.x += l_angular_velocity.y * r_xz * std::sin(theta);
	velocity.z += - l_angular_velocity.y * r_xz * std::cos(theta);


	return velocity;
}


void Model::initModel(int nParticles){
  /**
   * Compute the radiue of the Fe core and Si Shell
   * Known factors
   * 1. Fe core composes of 30% and the shell constitutes the rest
   * 2. Radius of earth
   * */
  h_particles = (Particle*)malloc(nParticles * sizeof(Particle));
  float g_radius_core_fe = modelParameters.g_radius_collidor * std::cbrt(0.3);
  int per_collidor_particles = (int) nParticles / 2;

  unsigned int num_iron_particles = (unsigned int) (modelParameters.PERCENT_IRON * per_collidor_particles);


  // initialize iron particle positions
  float3f rho;
  unsigned int i = 0;

  for(i = 0; i< num_iron_particles; i++){
    rho = rndFloat3();
    float3f position = randomSphere(rho, g_radius_core_fe);

    position = float3f(position.x + modelParameters.g_center_mass_one.x,
                        position.y + modelParameters.g_center_mass_one.y,
                        position.z + modelParameters.g_center_mass_one.z);
    float3f velocity = computeVelocity(position, CollidorEnum::ONE);
    h_particles[i] = Particle(position ,float3f(1.0f,0.0f,0.0f), velocity , ParticleType::IRON);
  }
  for(; i< per_collidor_particles;i++){
      rho = rndFloat3();
      float3f position = randomShell(rho, g_radius_core_fe);

      position = float3f(position.x + modelParameters.g_center_mass_one.x,
                          position.y + modelParameters.g_center_mass_one.y,
                          position.z + modelParameters.g_center_mass_one.z);
      float3f velocity = computeVelocity(position, CollidorEnum::ONE);
      h_particles[i] = Particle(position, float3f(1.0f,1.0f,1.0f), velocity , ParticleType::SILICA);
  }



  for(; i< per_collidor_particles + num_iron_particles; i++){
    rho = rndFloat3();
    float3f position = randomSphere(rho, g_radius_core_fe);
    position = float3f(position.x + modelParameters.g_center_mass_two.x,
                        position.y + modelParameters.g_center_mass_two.y,
                        position.z + modelParameters.g_center_mass_two.z);
    float3f velocity = computeVelocity(position, CollidorEnum::TWO);
    h_particles[i] = Particle(position , float3f(0.0f,1.0f,0.0f),velocity , ParticleType::IRON);
  }
  for(; i< nParticles;i++){
      rho = rndFloat3();
      float3f position = randomShell(rho, g_radius_core_fe);

      position = float3f(position.x + modelParameters.g_center_mass_two.x,
                          position.y + modelParameters.g_center_mass_two.y,
                          position.z + modelParameters.g_center_mass_two.z);
      float3f velocity = computeVelocity(position, CollidorEnum::TWO);
      h_particles[i] = Particle(position, float3f(1.0f,1.0f,1.0f),velocity , ParticleType::SILICA);
  }

}


void Model::releaseMemory(){
  free(h_particles);
}


Model::~Model(){

}
