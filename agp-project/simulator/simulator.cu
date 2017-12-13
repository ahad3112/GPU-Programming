#include<iostream>
#include <string>
#include <random>
#include <pthread.h>
#include "simulator.hpp"
#include "utilities.hpp"
#include "renderer.hpp"
#include "common.hpp"
#include "device.hpp"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);


using namespace glt;


// global variable . Particle lists
Particle* h_particles;
enum class CollidorEnum { ONE, TWO };
unsigned int NUM_PARTICLES 	= 100;
 const float PI = 3.1415927;                             // value of PI
float g_radius_collidor = 25576.86545f;
float3f g_center_mass_one = float3f(23925.0f,0.0f,9042.70f);
float3f g_center_mass_two = float3f(-23925.0f,0.0f,-9042.70f);

float3f g_linear_velocity_one = float3f(-3.24160f,0.0f,0.0f);
float3f g_linear_velocity_two = float3f(3.24160f,0.0f,0.0f);

float3f g_angular_velocity_one = float3f(0.0f,8.6036e-4f,0.0f);
float3f g_angular_velocity_two = float3f(0.0f,-8.6036e-4f,0.0f);
float PERCENT_IRON = 0.30f;                             // percentage of iron in a collidor
//float g_radius_earth  = 6371;
float g_radius_core_fe;
/*
* Forward method decleration
*/
void init(Particle* h_particles);
void printParticles(Particle* h_particles, int n);

/*
void *createGLFWwindow(void *threadID){
std::cout<<"Window thread started ... "<<std::endl;
window = utilities::createWindow("My Window...");
std::cout<<"Window sh created ... "<<std::endl;

 std::cout<<"Created shader program"<<std::endl;
 return NULL;
}
*/
// This method creates the glfw windiw and start
void *startRendering(void *threadID){
  std::cout<<"Rendering thread started ... "<<std::endl;
  render(h_particles,N);
  std::cout<<"Renered complete."<<std::endl;
  pthread_exit(NULL);
}






/*
* This method start the simulation
*/
void nbody::startSimulation(){
  std::cout<<"Simulation started..."<<std::endl;
  // allocate pinned memory for the host and initialize the Particle list
  h_particles = allocateHostMemory(N);
  init(h_particles);

  render(h_particles,N);
  /*
  createGLFWwindow();
  if(window){
    std::cout<<"Window is not null"<<std::endl;
  }
  */
/*
  pthread_t windowThread;
  int t = pthread_create(&windowThread,NULL,createGLFWwindow,NULL);
  pthread_join(windowThread,NULL);

  if(window){
    std::cout<<"Window is not null"<<std::endl;
  }
*/
/*
  pthread_t renderThread;
  int t1 = pthread_create(&renderThread,NULL,startRendering,NULL);
  // Initialize particles list
  printParticles(h_particles, 4);
  // Call cuda for initialization and execution of the kernel
  startComputation(h_particles,N);


  // Create another thread for glfw


  printParticles(h_particles, 4);

  pthread_join(renderThread,NULL);
  */
  // Release host memory
  releaseHostMemory(h_particles);

}



// This method print the list of initialized h_particles
void printParticles(Particle* h_particles, int n){
    for(int i = 0; i < n; i++){

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
float3f randomSphere(float3f rho){

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
float3f randomShell(float3f rho){

    float3f res;
    float pre,suf;

    float mu    = (1 - 2 * rho.y);
    float angle = 2*PI*rho.z;

    pre     = std::cbrt(std::pow(g_radius_core_fe,3) + (std::pow(g_radius_collidor,3) - std::pow(g_radius_core_fe,3)) * rho.x);
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
	theta = std::atan2((position.z - l_center_mass.z),(position.x - l_center_mass.x));
	velocity.x += l_angular_velocity.y * r_xz * std::sin(theta);
	velocity.z += - l_angular_velocity.z * r_xz * std::cos(theta);

	return velocity;
}

/**
 * This method initialized the list of particles (random for now...!!!!!!!!)
 */
void init(Particle* h_particles){
  // malloc should be replaced for page-locked memory

  /**
   * Compute the radiue of the Fe core and Si Shell
   * Known factors
   * 1. Fe core composes of 30% and the shell constitutes the rest
   * 2. Radius of earth
   * */

  g_radius_core_fe = g_radius_collidor * std::cbrt(0.3);
  int per_collidor_particles = (int) N / 2;

  unsigned int num_iron_particles = (unsigned int) (PERCENT_IRON * per_collidor_particles);
  //unsigned int num_silica_particles = NUM_PARTICLES - num_iron_particles;

  // initialize iron particle positions
  float3f rho;
  unsigned int i=0;

  for(i = 0; i< num_iron_particles; i++){
    rho = rndFloat3();
    float3f position = randomSphere(rho);
    //position = position + g_center_mass_one;
    position = float3f(position.x + g_center_mass_one.x,
                        position.y + g_center_mass_one.y,
                        position.z + g_center_mass_one.z);
    float3f velocity = computeVelocity(position, CollidorEnum::ONE);
    h_particles[i] = Particle(position , velocity , ParticleType::IRON);
  }
  for(; i< per_collidor_particles;i++){
      rho = rndFloat3();
      float3f position = randomShell(rho);
      //For particle one center of mass
      //position = position + g_center_mass_one;
      position = float3f(position.x + g_center_mass_one.x,
                          position.y + g_center_mass_one.y,
                          position.z + g_center_mass_one.z);
      float3f velocity = computeVelocity(position, CollidorEnum::ONE);
      h_particles[i] = Particle(position, velocity , ParticleType::SILICA);
  }



  for(; i< per_collidor_particles + num_iron_particles; i++){
    rho = rndFloat3();
    float3f position = randomSphere(rho);
    //position = position + g_center_mass_two;
    position = float3f(position.x + g_center_mass_two.x,
                        position.y + g_center_mass_two.y,
                        position.z + g_center_mass_two.z);
    float3f velocity = computeVelocity(position, CollidorEnum::TWO);
    h_particles[i] = Particle(position , velocity , ParticleType::IRON);
  }
  for(; i< N;i++){
      rho = rndFloat3();
      float3f position = randomShell(rho);
      //For particle two center of mass
      //position = position + g_center_mass_two;
      position = float3f(position.x + g_center_mass_two.x,
                          position.y + g_center_mass_two.y,
                          position.z + g_center_mass_two.z);
      float3f velocity = computeVelocity(position, CollidorEnum::TWO);
      h_particles[i] = Particle(position, velocity , ParticleType::SILICA);
  }

}
