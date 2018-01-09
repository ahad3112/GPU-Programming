#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include "simulator.hpp"
#include "particle.hpp"
#include "model.hpp"
#include "device.hpp"
#include "renderer.hpp"

#define WRITE_DATA 0
#define MAX_IT 1000


// This file is for writing particle position in each interaction
#if WRITE_DATA
  std::ofstream output;
#endif

void writeToFile(int N, Particle* h_particles){
  for(int i = 0; i < N ; i++){
    #if WRITE_DATA
      output<<h_particles[i].position.x<<" "<<h_particles[i].position.y<<"  "
      <<h_particles[i].position.z<<std::endl;
    #endif
  }
}

/*
* This method start the simulation
*/

void nbody::startSimulation(){
  std::cout<<"Simulation started..."<<std::endl;

  // opening file
  #if WRITE_DATA
    output.open("result.txt",std::ios_base::app);
  #endif

  // Get the model case to simulate
  Model* modelCase = new Model(3200,1); // case: 1,2,3

  // printf("nParticles: %d\n",modelCase->modelParameters.nParticles);
  // printf("g_linear_velocity_one: %f\n",modelCase->modelParameters.g_linear_velocity_one.x);

  // Get the renderer object to render in opengl
  Renderer renderer = Renderer(1280,720, modelCase->modelParameters.nParticles, modelCase->h_particles);

  // Get the device object to perform computation in cuda
  ComputeDevice computeDevice = ComputeDevice(renderer.resource,modelCase->h_particles, modelCase->modelParameters);

  // write to file
  writeToFile(modelCase->modelParameters.nParticles,modelCase->h_particles);
  int it = 0;
  while (!glfwWindowShouldClose(renderer.window)){
    printf("Iteration : %d\n",it++);
    // Perform computation in the device
    computeDevice.compute();
    // render the result
    renderer.display(modelCase->modelParameters.nParticles,renderer.window);

    #if WRITE_DATA
      // copy data from device to host
      cudaMemcpy(modelCase->h_particles, computeDevice.dev_particles, modelCase->modelParameters.nParticles * sizeof(Particle),cudaMemcpyDeviceToHost);
      // write data to file
      writeToFile(modelCase->modelParameters.nParticles,modelCase->h_particles);
    #endif

    if(it >= MAX_IT) break;

  }
  // Need to make sure to free the memory
  computeDevice.releaseMemory();
  renderer.releaseMemory();
  modelCase->releaseMemory();
}
