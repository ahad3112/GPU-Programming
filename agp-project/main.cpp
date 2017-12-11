
//#include "common.hpp"
#include "sphere.hpp"
#include "utilities.hpp"
#include "util.hpp"
#include <iostream>
#include <random>

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

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);


enum class ParticleType { IRON, SILICA };
enum class CollidorEnum { ONE, TWO};
/**
 *
 * Global variables
 * (Supplementary Table 1| Parameters for Planar single-tailed collision)
 * Unit of Weight : Kg
 * Unit of Distance : Km
 * Unit of Time : s
 */
unsigned int NUM_PARTICLES 	= 10000;	        // total no. of particles
unsigned int NUM_IT	        = 100;			// total number of iteration
float PERCENT_IRON          = 0.30f;        // percentage of iron in a collidor
float g_diameter 	= 376.78f;
float g_mass_si		= 7.4161e+19f;
float g_mass_fe		= 1.9549e+20f;
float g_k_si		= 2.9114e+14f;
float g_k_fe		= 5.8228e+14f;
float g_reduce_k_si	= 0.01f;
float g_reduce_k_fe	= 0.02f;
float g_sh_depth_si = 0.001f;
float g_sh_depth_fe = 0.002f;
float g_epsilon		= 47.0975f;
float g_time_step	= 5.8117f;


/**
 * Other constants
 * (Sourced from Internet)
 * */
//float g_radius_earth  = 6371;
float g_radius_earth  = 6371;
float g_radius_core_fe;


//For binding CUDA to OpenGL
Window                  win;
GLXContext              glc;
Display                 *dpy;
XVisualInfo             *vi;
GLint                   att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

struct floats {

  float x;
  float y;
  float z;
  floats(float x=0.0f,float y=0.0f, float z=0.0f): x(x),y(y),z(z) {}
  floats operator+(floats operand) { return floats(x+operand.x,y+operand.y,z+operand.z); }

};

typedef floats float3;

// Simulation settings
struct particleStruct{
  glm::vec3 position;
  glm::vec3 velocity;
  ParticleType pType;
  float radius;
  particleStruct(float3 pos, float3 velo, ParticleType p){
    position = glm::vec3(pos.x, pos.y, pos.z);
    velocity = glm::vec3(velo.x, velo.y, velo.z);
    pType = p;
  }
};

typedef particleStruct Particle;

glm::mat4 model, view, projection;
GLuint g_default_vao = 0;
unsigned int shaderProgram;
Particle* particles;
Particle* particles2;

// variable related to user interaction
float thetaZ = 0;
float circleRadius = 90000.0f;
float fov = 45.0f;



float3 g_center_mass_one = float3(23925.0f,0.0f,9042.7f);
float3 g_center_mass_two = float3(-23925.0f,0.0f,-9042.7f);
float3 g_linear_velocity_one = float3(-3.2416f,0.0f,0.0f);
float3 g_linear_velocity_two = float3(3.2416f,0.0f,0.0f);
float3 g_angular_velocity_one = float3(0.0f,8.6036e-4f,0.0f);
float3 g_angular_velocity_two = float3(0.0f,-8.6036e-4f,0.0f);

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


float3 rndFloat3(){
	return float3(rndFloat(),rndFloat(),rndFloat());
}

/**
 * Uses a random spherical distribution to create random locations in a sphere
 * returns : a random position in the sphere
 * */
float3 randomSphere(float3 rho){

    float3 res;
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
float3 randomShell(float3 rho){

    float3 res;
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


float3 computeVelocity(float3 position, CollidorEnum value){

	float3 l_center_mass;
	float3 l_linear_velocity;
	float3 l_angular_velocity;
	float3 velocity;
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
 **/
Particle* generateNewParticles(){
	// malloc should be replaced for page-locked memory
	//Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES); // non-pinned memory

    /**
     * Compute the radiue of the Fe core and Si Shell
     * Known factors
     * 1. Fe core composes of 30% and the shell constitutes the rest
     * 2. Radius of earth
     * */

    g_radius_core_fe = g_radius_earth * std::cbrt(0.3);

	//Particle* particles_impone;
	// Particle* particles_imptwo;
	// page-locked (pinned) memory
	//CUDA_CHECK_RETURN(cudaHostAlloc((void**)&particles, NUM_PARTICLES * sizeof(Particle) , cudaHostAllocDefault));

  Particle* particles_impone = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES);

    int num_iron_particles = (int) (PERCENT_IRON * NUM_PARTICLES);
    int num_silica_particles = NUM_PARTICLES - num_iron_particles;

    // initialize iron particle positions
    float3 rho;
    int i=0;
	for(i = 0; i< num_iron_particles; i++){
    rho = rndFloat3();
		float3 position = randomSphere(rho);
		position = position + g_center_mass_one;
		float3 velocity = computeVelocity(position, CollidorEnum::ONE);
		particles_impone[i] = Particle(position , velocity , ParticleType::IRON);
    }
    for(; i< NUM_PARTICLES;i++){
        rho = rndFloat3();
		float3 position = randomShell(rho);
    //For particle one center of mass
    position = position + g_center_mass_one;
		float3 velocity = computeVelocity(position, CollidorEnum::ONE);
        particles_impone[i] = Particle(position, velocity , ParticleType::SILICA);
    }
	return particles_impone;
}



//This method is the same as above for the second particle

Particle* generateNewParticles2(){
	// malloc should be replaced for page-locked memory
	//Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES); // non-pinned memory

    /**
     * Compute the radiue of the Fe core and Si Shell
     * Known factors
     * 1. Fe core composes of 30% and the shell constitutes the rest
     * 2. Radius of earth
     * */

    g_radius_core_fe = g_radius_earth * std::cbrt(0.3);

	//Particle* particles_impone;
	// Particle* particles_imptwo;
	// page-locked (pinned) memory
	//CUDA_CHECK_RETURN(cudaHostAlloc((void**)&particles, NUM_PARTICLES * sizeof(Particle) , cudaHostAllocDefault));

  Particle* particles_imptwo = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES);

    int num_iron_particles = (int) (PERCENT_IRON * NUM_PARTICLES);
    int num_silica_particles = NUM_PARTICLES - num_iron_particles;

    // initialize iron particle positions
    float3 rho;
    int i=0;
	for(i = 0; i< num_iron_particles; i++){
    rho = rndFloat3();
		float3 position = randomSphere(rho);
		position = position + g_center_mass_two;
		float3 velocity = computeVelocity(position, CollidorEnum::TWO);
		particles_imptwo[i] = Particle(position , velocity , ParticleType::IRON);
    }
    for(; i< NUM_PARTICLES;i++){
        rho = rndFloat3();
		float3 position = randomShell(rho);
    //For particle two center of mass
    position = position + g_center_mass_two;
		float3 velocity = computeVelocity(position, CollidorEnum::TWO);
        particles_imptwo[i] = Particle(position, velocity , ParticleType::SILICA);
    }
	return particles_imptwo;
}


// this method initialize and return a list of particles
Particle* generateParticles(int n){
  Particle* particles = (Particle*)malloc(sizeof(Particle) * n);
  for(int i = 0; i < n; i++){
    particles[i].position = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX));
    particles[i].velocity = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX));
    //particles[i].radius = getRND(.25f,0.3f);
    particles[i].radius = .10f;
  }
  return particles;
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
    view  = glm::lookAt(glm::vec3(0.0f, 0.0f, 60000.0f),
      glm::vec3(0.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 1.0f, 0.0f));
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 1000.0f, 100000.0f);
}

// This method release all allocated memory
void release()
{
    // Release the default VAO
    glDeleteVertexArrays(1, &g_default_vao);

    //Releasing CUDA

    //cuGLRegisterBufferObject( bufferID );
    // Rele the memory allocated for particle list
    free(particles);
    free(particles2);
}

void display(GLFWwindow *window)
{
        // uncomment this call to draw in wireframe polygons.
        //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        // render
        //glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // need to activate the shader before any calls to glUniform
        glUseProgram(shaderProgram);
            // Identity matrix
        //printf("%f %f %f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
        //model = glm::translate(model, glm::vec3(100000.0f,100.0f,100.0f));

        for(int i = 0; i< NUM_PARTICLES;i++){
          // create transformation matrices+glm::translate(model, particles2[i].position)
          // model = glm::translate(model, particles2[i].position);
          //model = glm::translate(model, glm::vec3(0.0f,0.0f,0.0f));
          // retrieve the matrix uniform locations
          model = glm::mat4(1.0f);
          model = glm::translate(model,particles[i].position);

          unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
          unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
          unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

          unsigned int FragColor = glGetUniformLocation(shaderProgram, "fragmentColor");
          // pass them to the shaders
          glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
          glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
          glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

          if(particles[i].pType ==ParticleType::IRON)
          {

              glUniform4f(FragColor,1.0f,0.0f,0.0f,0.2f);

          }
          else if(particles[i].pType ==ParticleType::SILICA)
          {
              glUniform4f(FragColor,0.0f,0.0f,1.0f,0.2f);


          }


          // draw our transformed particle as a sphere
          glutSolidSphere(80.8f,10,10);
          //glutWireSphere(particles[i].radius,9,9);

          // // update position randomly
          // particles[i].velocity = particles[i].velocity + glm::vec3(0.0f,-dt * g, 0.0f);
          particles[i].position = particles[i].position + particles[i].velocity * 2.0f;
          //particles2[i].position = particles2[i].position + particles2[i].velocity * 4.0f;


          // //glm::vec3 perturbation = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX)) * SCALE;
          // //particles[i].position = particles[i].position + perturbation;
          //
          // // Handling boundary condition
          // bool hitBoundary = false;
          // if(particles[i].position.x + particles[i].radius <= MIN || particles[i].position.x + particles[i].radius>= MAX){
          //   particles[i].position.x = glm::min(glm::max(particles[i].position.x,MIN),MAX);
          //   hitBoundary = true;
          // }
          // if(particles[i].position.y + particles[i].radius<= MIN || particles[i].position.y + particles[i].radius>= MAX){
          //   particles[i].position.y = glm::min(glm::max(particles[i].position.y,MIN),MAX);
          //   hitBoundary = true;
          // }
          // if(particles[i].position.z + particles[i].radius <= MIN || particles[i].position.z + particles[i].radius >= MAX){
          //   particles[i].position.z = glm::min(glm::max(particles[i].position.z,MIN),MAX);
          //   hitBoundary = true;
          // }
          //
          // if(hitBoundary){
          //   particles[i].velocity *= -1.0f;
          // }

          //particles[i].position.x = glm::min(glm::max(particles[i].position.x,MIN),MAX);
          //particles[i].position.y = glm::min(glm::max(particles[i].position.y,MIN),MAX);
          //particles[i].position.z = glm::min(glm::max(particles[i].position.z,MIN),MAX);
      }

      //For the second particle rendering
      for(int i = 0; i< NUM_PARTICLES;i++){
        // create transformation matrices+glm::translate(model, particles2[i].position)
        // model = glm::translate(model, particles2[i].position);
        //model = glm::translate(model, glm::vec3(0.0f,0.0f,0.0f));
        // retrieve the matrix uniform locations
        model = glm::mat4(1.0f);
        model = glm::translate(model,particles2[i].position);


        unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
        unsigned int viewLoc  = glGetUniformLocation(shaderProgram, "view");
        unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");

        unsigned int FragColor = glGetUniformLocation(shaderProgram, "fragmentColor");
        // pass them to the shaders
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

        if(particles2[i].pType ==ParticleType::IRON)
        {

            glUniform4f(FragColor,1.0f,0.0f,0.0f,0.2f);

        }
        else if(particles2[i].pType ==ParticleType::SILICA)
        {
            glUniform4f(FragColor,0.0f,0.0f,1.0f,0.2f);


        }


        // draw our transformed particle as a sphere
        glutSolidSphere(80.8f,10,10);
        //glutWireSphere(particles[i].radius,9,9);




        // // update position randomly
        // particles[i].velocity = particles[i].velocity + glm::vec3(0.0f,-dt * g, 0.0f);
        //particles[i].position = particles[i].position + particles[i].velocity * 2.0f;
        particles2[i].position = particles2[i].position + particles2[i].velocity * 4.0f;


        // //glm::vec3 perturbation = glm::vec3(getRND(MIN,MAX),getRND(MIN,MAX),getRND(MIN,MAX)) * SCALE;
        // //particles[i].position = particles[i].position + perturbation;
        //
        // // Handling boundary condition
        // bool hitBoundary = false;
        // if(particles[i].position.x + particles[i].radius <= MIN || particles[i].position.x + particles[i].radius>= MAX){
        //   particles[i].position.x = glm::min(glm::max(particles[i].position.x,MIN),MAX);
        //   hitBoundary = true;
        // }
        // if(particles[i].position.y + particles[i].radius<= MIN || particles[i].position.y + particles[i].radius>= MAX){
        //   particles[i].position.y = glm::min(glm::max(particles[i].position.y,MIN),MAX);
        //   hitBoundary = true;
        // }
        // if(particles[i].position.z + particles[i].radius <= MIN || particles[i].position.z + particles[i].radius >= MAX){
        //   particles[i].position.z = glm::min(glm::max(particles[i].position.z,MIN),MAX);
        //   hitBoundary = true;
        // }
        //
        // if(hitBoundary){
        //   particles[i].velocity *= -1.0f;
        // }

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
    thetaZ -= 1.0f;
    view = glm::lookAt(glm::vec3(circleRadius*sin(glm::radians(thetaZ)), 0.0f, circleRadius*cos(glm::radians(thetaZ))),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 1.0f, 0.0f));
  } else if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS){
    thetaZ += 1.0f;
    view = glm::lookAt(glm::vec3(circleRadius*sin(glm::radians(thetaZ)), 0.0f, circleRadius*cos(glm::radians(thetaZ))),
              glm::vec3(0.0f, 0.0f, 0.0f),
              glm::vec3(0.0f, 1.0f, 0.0f));
  }else if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS){
    fov -= 1.0f;
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 1000.0f, 100000.0f);
  } else if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS){
    fov += 1.0f;
    projection = glm::perspective(glm::radians(fov), (float)WIDTH / (float)HEIGHT, 1000.0f, 100000.0f);
  }
}

// Entry point of the program
int main()
{

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

		dpy = XOpenDisplay(NULL);

    if(dpy == NULL) {
        printf("\n\tcannot connect to X server\n\n");
        exit(0);
      }

    vi = glXChooseVisual(dpy, 0, att);
		glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);



    //CUDA-OpenGL binding part

	/*	glXMakeCurrent(dpy, win, glc);
		glEnable(GL_DEPTH_TEST);


//Creating OpenGL buffers

		GLuint bufferID;
// Generate a buffer ID
		glGenBuffers(1,&bufferID);
// Make this the current UNPACK buffer (OpenGL is state-based)

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	// Allocate data for the buffer
		glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, NULL,GL_DYNAMIC_COPY);

    cudaGLRegisterBufferObject( bufferID );

*/
    //map_buffer(bufferID);
    framebuffer_size_callback(window,WIDTH,HEIGHT);




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
  //std::cout<<"Entered here once!!";
  glViewport(0, 0, width, height);
  glLoadIdentity();
  glOrtho(0,1.0f,0,1.0f,-1.0f,1.0f);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

/*##############################################From the old git################################################### */

// This methods return a random float number


// float3 computeVelocity(float3 position, CollidorEnum value){
//
// 	float3 l_center_mass;
// 	float3 l_linear_velocity;
// 	float3 l_angular_velocity;
// 	float3 velocity;
// 	float r_xz;
// 	float theta;
// 	if(value == CollidorEnum::ONE){
// 		 l_center_mass = g_center_mass_one;
// 		 l_linear_velocity = g_linear_velocity_one;
// 		 l_angular_velocity = g_angular_velocity_one;
// 	}else{
// 		 l_center_mass = g_center_mass_two;
// 		 l_linear_velocity = g_linear_velocity_two;
// 		 l_angular_velocity = g_angular_velocity_two;
// 	}
//
// 	velocity = l_linear_velocity;
// 	r_xz = std::sqrt(pow((position.x - l_center_mass.x),2) + pow((position.z - l_center_mass.z),2));
// 	theta = atan2((position.z - l_center_mass.z)/(position.x - l_center_mass.x));
// 	velocity.x += l_angular_velocity.y * r_xz * std::sin(theta);
// 	velocity.y += - l_angular_velocity.y * r_xz * std::cos(theta);
//
// 	return velocity;
// }
//
// /**
//  * This method initialized the list of particles (random for now...!!!!!!!!)
//  **/
// Particle* init(){
// 	// malloc should be replaced for page-locked memory
// 	//Particle* particles = (Particle*)malloc(sizeof(Particle) * NUM_PARTICLES); // non-pinned memory
//
//     /**
//      * Compute the radiue of the Fe core and Si Shell
//      * Known factors
//      * 1. Fe core composes of 30% and the shell constitutes the rest
//      * 2. Radius of earth
//      * */
//
//     g_radius_core_fe = g_radius_earth * std::cbrt(0.3);
//
// 	Particle* particles_impone;
// 	// Particle* particles_imptwo;
// 	// page-locked (pinned) memory
// 	CUDA_CHECK_RETURN(cudaHostAlloc((void**)&particles, NUM_PARTICLES * sizeof(Particle) , cudaHostAllocDefault));
//
//     int num_iron_particles = (int) (PERCENT_IRON * NUM_PARTICLES);
//     int num_silica_particles = NUM_PARTICLES - num_iron_particles;
//
//     // initialize iron particle positions
//     float3 rho;
//     int i=0;
// 	for(i = 0; i< num_iron_particles; i++){
//         rho = rndFloat3();
// 		float3 position = randomSphere(rho);
// 		position = position + g_center_mass_one;
// 		float3 velocity = computeVelocity(position, CollidorEnum::ONE);
// 		particles_impone[i] = Particle(position , velocity , ParticleType::IRON);
//     }
//     for(; i< NUM_PARTICLES;i++){
//         rho = rndFloat3();
// 		float3 position = randomShell(rho);
// 		position = position + g_center_mass_one;
// 		float3 velocity = computeVelocity(position, CollidorEnum::ONE);
//         particles_impone[i] = Particle(position, velocity , ParticleType::SILICA);
//     }
// 	return particles;
// }
