#include <iostream>
#include <pthread.h>
#include "renderer.hpp"

#include "sphere.hpp"
#include "utilities.hpp"
#include "util.hpp"
#include "device.hpp"


#define MIN -10.0f
#define MAX 10.0f
#define g 9.80f
#define dt .002f


using namespace std;
using namespace glm;
using namespace glt;
using namespace agp;
using namespace agp::glut;


//OpenGL context for CUDA
float PI = 3.1415927;
unsigned int shaderProgram;
Particle* particles;
int device_N;





//For binding CUDA to OpenGL
Window                  win;
GLXContext              glc;
Display                 *dpy;
XVisualInfo             *vi;
GLint                   att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };


glm::mat4 model, view, projection;
GLuint g_default_vao = 0;
//unsigned int shaderProgram;

// variable related to user interaction
float thetaZ = 0;
float PARTICLE_ZOOM         = 3.5f;           // for scaling
float CAM_RADIUS = 105000.0f;
float fov = 45.0f;
float MIN_ZOOM = 5000.0f;
float MAX_ZOOM = 9000000.0f;
float ZOOM_STEP = 1000.0f;
float TIME_STEP = 4.0f;

// THis method return glm::vec3 from float3f
glm::vec3 getGlmVec(float3f v){
  return glm::vec3(v.x,v.y,v.z);
}

// This method initialize the 3D viewglfw3
void init()
{
    // Generate and bind the default VAO
    glGenVertexArrays(1, &g_default_vao);
    glBindVertexArray(g_default_vao);

    // Set the background color (RGBA)
    //glClearColor(0.0f, 1.0f, 0.0f, 0.0f);

    // Your OpenGL settings, such as alpha, depth and others, should be
    // defined here! For the assignment, we only ask you to enable the
    // alpha channel.
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // initial view and projection matrices
    view  = glm::lookAt(glm::vec3(0.0f,CAM_RADIUS, 0.0f),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 1.0f));
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 1000.0f, 1000000.0f);
}

// This method release all allocated memory
void release()
{
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);
}

void display(GLFWwindow *window,Particle* h_particles, int n)
{


        // uncomment this call to draw in wireframe polygons.
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // render
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // need to activate the shader before any calls to glUniform
        glUseProgram(shaderProgram);



        for(uint i = 0; i< n ;i++){
          // create transformation matrices+glm::translate(model, particles2[i].position)
          // model = glm::translate(model, particles2[i].position);
          //model = glm::translate(model, glm::vec3(0.0f,0.0f,0.0f));
          // retrieve the matrix uniform locations
          model = glm::mat4(1.0f);
          model = glm::scale(glm::translate(model,getGlmVec(h_particles[i].position)),glm::vec3( PARTICLE_ZOOM ));

          unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
          unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
          unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

          unsigned int FragColor = glGetUniformLocation(shaderProgram, "fragmentColor");
          // pass them to the shaders
          glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
          glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
          glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

          if(h_particles[i].pType == ParticleType::IRON)
          {

              glUniform4f(FragColor,1.0f,0.0f,0.0f,0.2f);

          }
          else if(h_particles[i].pType == ParticleType::SILICA)
          {
              glUniform4f(FragColor,0.0f,0.0f,1.0f,0.2f);


          }

          // draw our transformed particle as a sphere
          glutSolidSphere(80.8f,10,10);
          //glutWireSphere(particles[i].radius,9,9);

          // // update position randomly

          //particles[i].position = particles[i].position + particles[i].velocity * TIME_STEP;

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
    thetaZ -= 1.0f;
    view = glm::lookAt(glm::vec3(CAM_RADIUS*sin(glm::radians(thetaZ)),CAM_RADIUS*cos(glm::radians(thetaZ)), 0.0f ),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 0.0f, 1.0f));
  } else if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
    thetaZ += 1.0f;
    view = glm::lookAt(glm::vec3(CAM_RADIUS*sin(glm::radians(thetaZ)), CAM_RADIUS*cos(glm::radians(thetaZ)), 0.0f ),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 0.0f, 1.0f));
  }else if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){

    if(CAM_RADIUS-ZOOM_STEP <= MIN_ZOOM)
      CAM_RADIUS = MIN_ZOOM;
    else
      CAM_RADIUS -= ZOOM_STEP;
          view = glm::lookAt(glm::vec3(CAM_RADIUS*sin(glm::radians(thetaZ)),CAM_RADIUS*cos(glm::radians(thetaZ)), 0.0f ),
          glm::vec3(0.0f, 0.0f, 0.0f),
          glm::vec3(0.0f, 0.0f, 1.0f));
  } else if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
    if(CAM_RADIUS-ZOOM_STEP >= MAX_ZOOM)
      CAM_RADIUS = MAX_ZOOM;
    else
      CAM_RADIUS += ZOOM_STEP;
        view = glm::lookAt(glm::vec3(CAM_RADIUS*sin(glm::radians(thetaZ)),CAM_RADIUS*cos(glm::radians(thetaZ)), 0.0f ),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
  }
}

void *callCuda(void *threadID){
  std::cout<<"Rendering thread started ... "<<std::endl;
  startComputation(particles,device_N);
  std::cout<<"Renered complete."<<std::endl;
  pthread_exit(NULL);
}



void render(Particle* h_particles, int n){
  device_N = n;
  particles = h_particles;
  std::cout<<"Rendering in opengl started..."<<std::endl;
  GLFWwindow* window = utilities::createWindow("My Window...");

  // Create the shader program
 //unsigned int shaderProgram = util::createShaderProgram("/Users/maaahad/Documents/OpenGL_Projects/OpengGL_MVP/OpengGL_MVP/myShaders.glsl");
 shaderProgram = utilities::createShaderProgram("myShaders.glsl");

 glUseProgram(shaderProgram);
  std::cout<<"shader program created..."<<std::endl;
  glfwSetKeyCallback(window,key_callback);

// // device information
  util::displayOpenGLInfo();
 // Initialize 3d view
 init();


 // Start computation
 pthread_t cudaThread;
 int t1 = pthread_create(&cudaThread,NULL,callCuda,NULL);
 //Creating OpenGL context and binding it to window


//std::cout<<"("<<h_particles[0].positions.x<<","<<h_particles[0].positions.y<<"," <<h_particles[0].positions.z<<")"<<std::endl;
  while (!glfwWindowShouldClose(window)){
    //printf("(%f,%f,%f)\n", h_particles[0].positions.x, h_particles[0].positions.y, h_particles[0].positions.z);
    display(window,h_particles,n);
  }
  pthread_join(cudaThread,NULL);
  // Terminate the window
  glfwTerminate();
}
