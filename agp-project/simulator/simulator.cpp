#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <fstream>
#include "simulator.hpp"
#include "particle.hpp"
#include "model.hpp"
#include "device.hpp"
#include "renderer.hpp"

#define CPU_SIM 0               // set 0 for simulation in CUDA, anything except 0 will perform simulation in CPU
#define COPY_FROM_DEVICE 0        // in case we need to write the simulation data
#define WRITE_DATA 0              // in case we need to write the simulation data

#if CPU_SIM
#include "cpuSim.hpp"
#endif

/*
* Implementation of Simulator
*
*/

// This file is for writing particle position in each interaction
#if WRITE_DATA && COPY_FROM_DEVICE
  std::ofstream output;
#endif

void writeToFile(int N, Particle* h_particles){
  for(int i = 0; i < N ; i++){
    #if WRITE_DATA && COPY_FROM_DEVICE
      output<<h_particles[i].position.x<<" "<<h_particles[i].position.y<<"  "
      <<h_particles[i].position.z<<std::endl;
    #endif
  }
}


// Define renderer as a global variable to be used in key_callback methods
Renderer renderer;


// Setting callback method for glfw window to interact with the user
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
  if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
      std::cout<<"Esc has been pressed..."<<std::endl;
      glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
    renderer.thetaZ -= 1.0f;
    renderer.view = glm::lookAt(glm::vec3(renderer.CAM_RADIUS*sin(glm::radians(renderer.thetaZ)),renderer.CAM_RADIUS*cos(glm::radians(renderer.thetaZ)), 0.0f ),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 0.0f, 1.0f));
  } else if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
    renderer.thetaZ += 1.0f;
    renderer.view = glm::lookAt(glm::vec3(renderer.CAM_RADIUS*sin(glm::radians(renderer.thetaZ)), renderer.CAM_RADIUS*cos(glm::radians(renderer.thetaZ)), 0.0f ),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 0.0f, 1.0f));
  } else if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){

    if(renderer.CAM_RADIUS-renderer.ZOOM_STEP <= renderer.MIN_ZOOM)
      renderer.CAM_RADIUS = renderer.MIN_ZOOM;
    else
      renderer.CAM_RADIUS -= renderer.ZOOM_STEP;

      renderer.view = glm::lookAt(glm::vec3(renderer.CAM_RADIUS*sin(glm::radians(renderer.thetaZ)),renderer.CAM_RADIUS*cos(glm::radians(renderer.thetaZ)), 0.0f ),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 1.0f));
  } else if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
    if(renderer.CAM_RADIUS-renderer.ZOOM_STEP >= renderer.MAX_ZOOM)
      renderer.CAM_RADIUS = renderer.MAX_ZOOM;
    else
      renderer.CAM_RADIUS += renderer.ZOOM_STEP;

      renderer.view = glm::lookAt(glm::vec3(renderer.CAM_RADIUS*sin(glm::radians(renderer.thetaZ)),renderer.CAM_RADIUS*cos(glm::radians(renderer.thetaZ)), 0.0f ),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 1.0f));
  }

  // Update the mvp matrix
  renderer.mvp = renderer.projection * renderer.view * renderer.model;
  unsigned int mvpID = glGetUniformLocation(renderer.shaderProgram, "MVP");
  glUniformMatrix4fv(mvpID, 1, GL_FALSE, glm::value_ptr(renderer.mvp));
  glUseProgram(renderer.shaderProgram);

}

/*
* This method start the simulation
*/

void nbody::startSimulation(const int n, const int modelCase, const int max_it, const int width, const int height, const int BLOCK_SIZE){
  #if CPU_SIM
   std::cout<<"Simulation started in CPU."<<std::endl;
  #endif

  #if !CPU_SIM
   std::cout<<"Simulation started in GPU."<<std::endl;
  #endif

  // opening file
  #if WRITE_DATA && COPY_FROM_DEVICE
    output.open("result.txt",std::ios_base::app);
  #endif

  // Get the model case to simulate
  Model* model = new Model(n,modelCase); // case: 1,2,3

  // printf("nParticles: %d\n",model->modelParameters.nParticles);
  // printf("g_linear_velocity_one: %f\n",model->modelParameters.g_linear_velocity_one.x);

  // Get the renderer object to render in opengl and set the key-callback method
  renderer = Renderer(width,height, model->modelParameters.nParticles, model->h_particles);
  glfwSetKeyCallback(renderer.window, key_callback);


  // Get the device object to perform computation in cuda
  ComputeDevice computeDevice = ComputeDevice(renderer.resource,model->h_particles, model->modelParameters,BLOCK_SIZE);

  // write to file if required
  #if WRITE_DATA && COPY_FROM_DEVICE
    writeToFile(model->modelParameters.nParticles,model->h_particles);
  #endif

  // Variable to track the no. of fiterations
  int it = 0;

  // Starting cuda event to initiate record execution time
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  for(; it<max_it; it++){
    //printf("Iteration : %d\n",it++);

    // Perform simulation in the device
    #if !CPU_SIM
      // printf("GPU simulaiton.\n");
      computeDevice.compute();
    #endif
    // upnmap the shared resource. This works as synchronization between the cuda and graphics portion of the appllication
    //cudaGraphicsUnmapResources(1,&renderer.resource,NULL);

    // Perform simulation in CPU
    #if CPU_SIM
      // printf("CPU simulaiton.\n");
      updateEarthMoonSystemCPU(model->h_particles,&model->modelParameters);
      renderer.reBuffer(model->modelParameters.nParticles, model->h_particles);
    #endif

    // This is to check performance while we copy from device to host
    #if COPY_FROM_DEVICE
      // copy data from device to host
      cudaMemcpy(model->h_particles, computeDevice.dev_particles, model->modelParameters.nParticles * sizeof(Particle),cudaMemcpyDeviceToHost);
      renderer.reBuffer(model->modelParameters.nParticles, model->h_particles);
    #endif

    // render the result
    renderer.display(model->modelParameters.nParticles,renderer.window);


    #if WRITE_DATA && COPY_FROM_DEVICE
      // write data to file
      writeToFile(model->modelParameters.nParticles,model->h_particles);
    #endif

    // close the window and the simulation if ESC is pressed
    if(glfwWindowShouldClose(renderer.window)) break;


  }
  // record the end time of execution
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Simulation time for NUM_PARTICLES = %d , NUM_ITERATION = %d and Thread BLOCK_SIZE = %d : %3.1f ms \n",n,it,BLOCK_SIZE,elapsedTime);

  // free the allocated resource
  computeDevice.releaseMemory();
  renderer.releaseMemory();
  model->releaseMemory();
}
