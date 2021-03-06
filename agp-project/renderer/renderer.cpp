#include <iostream>
#include <stdio.h>
#include "utilities.hpp"
#include "util.hpp"
#include "renderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace glt;
using namespace agp;

/**
* Implementation of Renderer class
*
*/

Renderer::Renderer(){
}

Renderer::Renderer(int WIDTH, int HEIGHT, int nParticles, Particle* h_particles) : WIDTH(WIDTH), HEIGHT(HEIGHT){
  // // First we choose the cuda device to be used
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 3;
  prop.minor = 0;
  cudaChooseDevice(&dev, &prop);
  cudaGLSetGLDevice(dev);

  printf("Creating renderer...\n");
  window = utilities::createWindow("Earth-Moon System...", WIDTH, HEIGHT);
  shaderProgram = utilities::createShaderProgram("myShaders.glsl");
  glUseProgram(shaderProgram);

  // Set the key call back method
  //glfwSetKeyCallback(window, Renderer::key_callback);

  // device informationSSs
  printf("\n---- System Information Start ---- \n");
  util::displayOpenGLInfo();
  printf("---- System Information End ---- \n\n");
  // init opengl stuff
  initMVP();
  initGL(nParticles, h_particles);

}

/**
* this method Initialize Model, View, Projection matrix
*/
void Renderer::initMVP(){
  // initial view and projection matrices
  model = glm::mat4(1.0f);
  view  = glm::lookAt(glm::vec3(0.0f,CAM_RADIUS, 0.0f),
    glm::vec3(0.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 1.0f));
  projection = glm::perspective(
    glm::radians(fov),
    (float)WIDTH / (float)HEIGHT,
    NEAR_PLANE,
    FAR_PLANE
  );

  mvp = projection * view * model;
  unsigned int mvpID = glGetUniformLocation(shaderProgram, "MVP");
  glUniformMatrix4fv(mvpID, 1, GL_FALSE, glm::value_ptr(mvp));
  glUseProgram(shaderProgram);

}

/**
* Initialize OpenGL RELATED STUFF
*/
void Renderer::initGL(int nParticles, Particle* h_particles){
  // OpenGL settings, such as alpha, depth and others, should be
  // defined here! For the assignment, we only ask you to enable the
  // alpha channel.
  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE); // This method enable vertex size in vertex shader
  glClearColor(0.0f,0.0f,0.0f,1.0f);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // generate shared buffer and register it to use in cuda kernel
  glGenVertexArrays(1,&arrayObj);
  glGenBuffers(1, &bufferObj);
  glBindVertexArray(arrayObj);
  glBindBuffer(GL_ARRAY_BUFFER,bufferObj);
  glBufferData(GL_ARRAY_BUFFER, nParticles * sizeof(Particle),h_particles,GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Particle), (void*)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Particle), (void*) (sizeof(float3f)));

  // register bufferobj as cuda resource
  cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
}

/**
* This method display the simulation
*/
void Renderer::display(int nParticles,GLFWwindow *window){
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawArrays(GL_POINTS, 0,nParticles);
  glfwSwapBuffers(window);
  glfwPollEvents();
}

/**
* This method reBuffer particles list in case of CPU only simulation
*/
void Renderer::reBuffer(int nParticles, Particle* h_particles){
    glBufferData(GL_ARRAY_BUFFER, nParticles * sizeof(Particle),h_particles,GL_DYNAMIC_DRAW);
}
/**
* Release resource
*/
void Renderer::releaseMemory(){
  glDeleteVertexArrays(1, &arrayObj);
  glfwTerminate();
}
