#include<iostream>
#include <string>
#include "common.hpp"
#include "simulator.hpp"
#include "particle.hpp"
#include "device.hpp"

enum class CollidorEnum { ONE, TWO };

/*
* Forward method decleration
*/
void init(Particle* h_particles);
void printParticles(Particle* h_particles);
// This method initialied the list of particle

/*
* This method start the simulation
*/
void nbody::startSimulation(){
  std::cout<<"Simulation started..."<<std::endl;
  // allocate pinned memory for the host and initialize the Particle list
  Particle* h_particles = allocateHostMemory(NUM_PARTICLES);

  // Initialize particles list
  init(h_particles);
  printParticles(h_particles);
  // Call cuda for initialization and execution of the kernel
  startComputation(h_particles,NUM_PARTICLES);

  // Release host memory
  releaseHostMemory(h_particles);
}

// This method print the list of initialized h_particles
void printParticles(Particle* h_particles){
    for(int i = 0; i < NUM_PARTICLES; i++){

      std::cout<<"Particle: "<<i
              <<" -> ("<<
              h_particles[i].position.x
              <<" , "<<
              h_particles[i].position.y
              <<" , "<<
              h_particles[i].position.z
              <<" )"<<
              std::endl;
    }
}

// This methods return a random float number
float rndFloat(){
	return  static_cast <float>(rand()) / static_cast <float> (RAND_MAX);
}

// This methods return a random float number between two given floats
float getRND(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

float3f rndFloat3(){
	return float3f(rndFloat(),rndFloat(),rndFloat());
}

/**
 * Uses a random spherical distribution to create random locations in a sphere
 * returns : a random position in the sphere
 * */
float3f randomSphere(float3f rho){

    float3f res = float3f();
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
 * outer radius : g_radius_earth
 * returns : a random position in the shell
 * */
float3f randomShell(float3f rho){

    float3f res = float3f();
    float pre,suf;

    float mu    = (1 - 2 * rho.y);
    float angle = 2*PI*rho.z;

    pre     = std::cbrt(std::pow(g_radius_core_fe,3) + (std::pow(g_radius_earth,3) - std::pow(g_radius_core_fe,3)) * rho.x);
    suf     = std::sqrt(1 - std::pow(mu,2));
    res.x   = pre * suf * std::cos(angle);
    res.y   = pre * suf * std::sin(angle);
    res.z   = pre * mu;

    return res;

}


float3f computeVelocity(float3f position, CollidorEnum value){

	float3f l_center_mass;
	float3f l_linear_velocity;
	float3f l_angular_velocity;
	float3f velocity;

	float r_xz;
	float theta;
	if(value == CollidorEnum::ONE){
		 l_center_mass = g_center_mass_one;
		 l_linear_velocity = g_linear_velocity_one;
		 l_angular_velocity = g_angular_velocity_one;
	}else{
		 l_center_mass = g_center_mass_two;
		 l_linear_velocity = g_linear_velocity_two;
		 l_angular_velocity = g_angular_velocity_two;
	}

	velocity = l_linear_velocity;
	r_xz = std::sqrt(pow((position.x - l_center_mass.x),2) + pow((position.z - l_center_mass.z),2));
	theta = atan2((position.z - l_center_mass.z),(position.x - l_center_mass.x)); // Need to recheck //
	velocity.x += l_angular_velocity.y * r_xz * std::sin(theta);
	velocity.y += - l_angular_velocity.y * r_xz * std::cos(theta);

	return velocity;
}

/**
 * This method initialized the list of particles (random for now...!!!!!!!!)
 */
void init(Particle* h_particles){
	// malloc should be replaced for page-locked memory
	//Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES); // non-pinned memory

    /**
     * Compute the radiue of the Fe core and Si Shell
     * Known factors
     * 1. Fe core composes of 30% and the shell constitutes the rest
     * 2. Radius of earth
     * */

    g_radius_core_fe = g_radius_earth * std::cbrt(0.3);

    int per_collidor_particles = (int) NUM_PARTICLES / 2;
    int num_iron_particles = (int) (PERCENT_IRON * per_collidor_particles);
    int num_silica_particles = per_collidor_particles - num_iron_particles;

    // adding first collidor one
    float3f rho;
    int i = 0;

  	for(i = 0; i< num_iron_particles; i++){
          rho = rndFloat3();
//          std::cout<<"rho.x<<" "<<rho.y<<" "<<rho.z"<<std::endl;
//          std::cout<<rho.x<<" "<<rho.y<<" "<<rho.z<<std::endl;
      		float3f position = randomSphere(rho);

          position = float3f(position.x + g_center_mass_one.x,
                              position.y + g_center_mass_one.y,
                              position.z + g_center_mass_one.z);

      		float3f velocity = computeVelocity(position, CollidorEnum::ONE);
      		h_particles[i] = Particle(position , velocity , ParticleType::IRON);

      }

      for(; i< per_collidor_particles;i++){
          rho = rndFloat3();
      		float3f position = randomShell(rho);
          position = float3f(position.x + g_center_mass_one.x,
                              position.y + g_center_mass_one.y,
                              position.z + g_center_mass_one.z);

      		float3f velocity = computeVelocity(position, CollidorEnum::ONE);
          h_particles[i] = Particle(position, velocity , ParticleType::SILICA);

      }

      // adding second Collidor
      for(; i< per_collidor_particles + num_iron_particles; i++){
            rho = rndFloat3();
  //          std::cout<<"rho.x<<" "<<rho.y<<" "<<rho.z"<<std::endl;
  //          std::cout<<rho.x<<" "<<rho.y<<" "<<rho.z<<std::endl;
            float3f position = randomSphere(rho);

            position = float3f(position.x + g_center_mass_two.x,
                                position.y + g_center_mass_two.y,
                                position.z + g_center_mass_two.z);

            float3f velocity = computeVelocity(position, CollidorEnum::TWO);
            h_particles[i] = Particle(position , velocity , ParticleType::IRON);

        }

        for(; i< NUM_PARTICLES;i++){
            rho = rndFloat3();
            float3f position = randomShell(rho);
            position = float3f(position.x + g_center_mass_two.x,
                                position.y + g_center_mass_two.y,
                                position.z + g_center_mass_two.z);

            float3f velocity = computeVelocity(position, CollidorEnum::TWO);
            h_particles[i] = Particle(position, velocity , ParticleType::SILICA);
        }

}
