
//#include "common.hpp"
#include <iostream>
#include <random>
#include "sphere.hpp"
#include "utilities.hpp"
#include "util.hpp"
#include "simulator.hpp"


// Entry point of the program
int main()
{
  nbody::startSimulation();

  /*
  GLFWwindow* window = utilities::createWindow("My Window...");

  glfwSetKeyCallback(window,key_callback);

  // Create the shader program
//unsigned int shaderProgram = util::createShaderProgram("/Users/maaahad/Documents/OpenGL_Projects/OpengGL_MVP/OpengGL_MVP/myShaders.glsl");
shaderProgram = utilities::createShaderProgram("myShaders.glsl");

glUseProgram(shaderProgram);

// device information
util::displayOpenGLInfo();

// generate particles Array
particles = generateNewParticles();
particles2 = generateNewParticles2();

// Initialize 3d view
init();

//Creating OpenGL context and binding it to window

while (!glfwWindowShouldClose(window))
{
  display(window);

}

// Terminate the window
glfwTerminate();
*/
  return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
// void framebuffer_size_callback(GLFWwindow* window, int width, int height)
// {
//   // make sure the viewport matches the new window dimensions; note that width and
//   // height will be significantly larger than specified on retina displays.
//   //std::cout<<"Entered here once!!";
//   glViewport(0, 0, width, height);
//   glLoadIdentity();
//   glOrtho(0,1.0f,0,1.0f,-1.0f,1.0f);
//   glMatrixMode(GL_MODELVIEW);
//   glLoadIdentity();
// }
