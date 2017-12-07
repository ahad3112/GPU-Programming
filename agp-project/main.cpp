
#include "common.hpp"
#include "sphere.hpp"
#include "utilities.hpp"
#include "util.hpp"
#include <iostream>

#define MIN -10.0f
#define MAX 10.0f
#define g 9.80f
#define dt .002f


using namespace std;
using namespace glm;
using namespace glt;
using namespace agp;
using namespace agp::glut;


void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// Simulation settings
typedef struct {
  glm::vec3 position;
  glm::vec3 velocity;
  float radius;
}Particle;

glm::mat4 model, view, projection;
GLuint g_default_vao = 0;
unsigned int shaderProgram;
Particle* particles;

// variable related to user interaction
float thetaZ = 0;
float circleRadius = 5.0f;
float fov = 45.0f;

// This method return random float between the two given float
float getRND(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

// this method initialize and return a list of particles
Particle* generateParticles(int n){
  Particle* particles = (Particle*)malloc(sizeof(Particle) * n);
  for(int i = 0; i < n; i++){
    particles[i].position = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX));
    particles[i].velocity = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX));
    //particles[i].radius = getRND(.25f,0.3f);
    particles[i].radius = .50f;
  }
  return particles;
}

// This method initialize the 3D view
void init()
{
    // Generate and bind the default VAO
    glGenVertexArrays(1, &g_default_vao);
    glBindVertexArray(g_default_vao);

    // Set the background color (RGBA)
    glClearColor(0.0f, 1.0f, 0.0f, 0.0f);

    // Your OpenGL settings, such as alpha, depth and others, should be
    // defined here! For the assignment, we only ask you to enable the
    // alpha channel.
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // initial view and projection matrices
    view  = glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 1.0f, 0.0f));
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
}

// This method release all allocated memory
void release()
{
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);

    // Rele the memory allocated for particle list
    free(particles);
}

void display(GLFWwindow *window)
{
        // uncomment this call to draw in wireframe polygons.
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // render
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // need to activate the shader before any calls to glUniform
        glUseProgram(shaderProgram);


        for(int i = 0; i< N;i++){
          // create transformation matrices
          model = glm::mat4(1.0f);    // Identity matrix
          model = glm::translate(model, particles[i].position);

          // retrieve the matrix uniform locations
          unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
          unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
          unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
          // pass them to the shaders
          glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
          glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
          glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

          // draw our transformed particle as a sphere
          glutSolidSphere(particles[i].radius,30,30);
          glutWireSphere(particles[i].radius,30,30);

          // update position randomly
          particles[i].velocity = particles[i].velocity + glm::vec3(0.0f,-dt * g, 0.0f);
          particles[i].position = particles[i].position + particles[i].velocity * dt;
          //glm::vec3 perturbation = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX)) * SCALE;
          //particles[i].position = particles[i].position + perturbation;

          // Handling boundary condition
          bool hitBoundary = false;
          if(particles[i].position.x + particles[i].radius <= MIN || particles[i].position.x + particles[i].radius>= MAX){
            particles[i].position.x = glm::min(glm::max(particles[i].position.x,MIN),MAX);
            hitBoundary = true;
          }
          if(particles[i].position.y + particles[i].radius<= MIN || particles[i].position.y + particles[i].radius>= MAX){
            particles[i].position.y = glm::min(glm::max(particles[i].position.y,MIN),MAX);
            hitBoundary = true;
          }
          if(particles[i].position.z + particles[i].radius <= MIN || particles[i].position.z + particles[i].radius >= MAX){
            particles[i].position.z = glm::min(glm::max(particles[i].position.z,MIN),MAX);
            hitBoundary = true;
          }

          if(hitBoundary){
            particles[i].velocity *= -1.0f;
          }

          //particles[i].position.x = glm::min(glm::max(particles[i].position.x,MIN),MAX);
          //particles[i].position.y = glm::min(glm::max(particles[i].position.y,MIN),MAX);
          //particles[i].position.z = glm::min(glm::max(particles[i].position.z,MIN),MAX);
      }
      // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
      // -------------------------------------------------------------------------------
      glfwSwapBuffers(window);
      glfwPollEvents();
}

// key callback method
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
    glfwSetWindowShouldClose(window, true);
  } else if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS){
    thetaZ -= 10.0f;
    view = glm::lookAt(glm::vec3(circleRadius*sin(glm::radians(thetaZ)), 0.0f, circleRadius*cos(glm::radians(thetaZ))),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 1.0f, 0.0f));
  } else if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
    thetaZ += 10.0f;
    view = glm::lookAt(glm::vec3(circleRadius*sin(glm::radians(thetaZ)), 0.0f, circleRadius*cos(glm::radians(thetaZ))),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 1.0f, 0.0f));
  }else if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
    fov -= 5.0f;
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
  } else if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
    fov += 5.0f;
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 0.1f, 100.0f);
  }
}

// Entry point of the program
int main()
{
  GLFWwindow* window = utilities::createWindow("My Window...");

  // Create the shader program
//unsigned int shaderProgram = util::createShaderProgram("/Users/maaahad/Documents/OpenGL_Projects/OpengGL_MVP/OpengGL_MVP/myShaders.glsl");
shaderProgram = utilities::createShaderProgram("myShaders.glsl");

glUseProgram(shaderProgram);

// device information
util::displayOpenGLInfo();

// generate particles Array
particles = generateParticles(N);

// Initialize 3d view
init();

while (!glfwWindowShouldClose(window))
{
  display(window);

}

// Terminate the window
glfwTerminate();

  return 0;
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width and
  // height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}
