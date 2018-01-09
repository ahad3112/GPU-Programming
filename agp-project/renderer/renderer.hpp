#ifndef RENDERER_HPP
#define RENDERER_HPP
#include "particle.hpp"
#include <glad/glad.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Renderer {
private:
  // window WIDTH and HEIGHT
  int WIDTH;
  int HEIGHT;
  // variable related to user interaction
  float thetaZ = 0.0f;
  float PARTICLE_ZOOM         = 3.5f;           // for scaling
  float CAM_RADIUS = 105000.0f;
  float fov = 45.0f;
  float MIN_ZOOM = 5000.0f;
  float MAX_ZOOM = 9000000.0f;
  float ZOOM_STEP = 1000.0f;
  float TIME_STEP = 4.0f;
  float NEAR_PLANE = 1000.0f;
  float FAR_PLANE = 1000000.0f;

  // M,V,P matrix
  glm::mat4 model, view, projection, mvp;

  // Shader program
  unsigned int shaderProgram;



public:
  // buffer object to be used by the opengl
  GLuint bufferObj;
  GLuint arrayObj;
  cudaGraphicsResource* resource;
  // GLFW window
  GLFWwindow *window;
  Renderer();
  Renderer(int WIDTH, int HEIGHT, int nParticles, Particle* h_particles);
  void initMVP();
  void initGL(int nParticles,Particle* h_particles);
  void display(int nParticles,GLFWwindow *window);
  static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void interact();
  void releaseMemory();
};
#endif
