#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include "device.hpp"
#include "commonDevice.hpp"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Forward method decleration
*/
void writeToFile(int N, Particle* h_particles);
bool ovelapCapabled();
__device__ float3f interaction(Particle currentParticle, Particle otherParticle, float3f accel);

/**
* Implementation of float3f's methods
*/
  __host__ __device__ float3f::float3f(float x, float y, float z) : x(x), y(y), z(z){

  }

  __host__ __device__ float3f::float3f(){
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
  }


  __host__ __device__ float3f::~float3f(){

  }

  /**
  * Implementation of Particle's methods
  */
  __host__ __device__ Particle::Particle(float3f position, float3f velocity, ParticleType pType) :
  position(position), velocity(velocity), pType(pType)
  {
    //std::cout<<"Initializing particle..."<<std::endl;
  }

  __host__ __device__ Particle::~Particle(){

  }


/**
* This method allocate pinned memory for the particle initiaization in host
*/
Particle* allocateHostMemory(int N){
  Particle* h_particles;
  CUDA_CHECK_RETURN(cudaHostAlloc((void**) &h_particles, N * sizeof(Particle) , cudaHostAllocDefault));
  std::cout<<"Allocated pinned memory for the host."<<std::endl;
  return h_particles;
}
/**
* This method release pinned memory allocated for the particle initiaization in host
*/
void releaseHostMemory(Particle* h_particles){
  CUDA_CHECK_RETURN(cudaFreeHost(h_particles));
  std::cout<<"Deleted pinned memory from the host."<<std::endl;
}

// Cuda error check
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err){
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

__global__ void sampleKernel(){
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("Thread: %d\n", gtid);
}


// gpu kernel for position update
__global__ void updatePosition(int N, Particle* particles, float3f* accels){
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gtid < N){
    //printf("accels : %d : %f\n",gtid, accels[gtid].x);
		//printf("updatePosition : %d : %f\n",idx, particles[idx].position.x);
		// Need to add contribution from angular velocity as well
		// Numerical methods: Euler method. Update later to Leaf-frog method
		// update velocity
		float3f vel = particles[gtid].velocity;
		vel.x += accels[gtid].x * g_time_step;
		vel.y += accels[gtid].y * g_time_step;
		vel.z += accels[gtid].z * g_time_step;
		// update position
		float3f position = particles[gtid].position;
		position.x += vel.x * g_time_step;
		position.y += vel.y * g_time_step;
		position.z += vel.z * g_time_step;

		particles[gtid].velocity = vel;
		particles[gtid].position = position;

		//printf("p%d: (%f,%f,%f)\n",gtid,particles[gtid].position.x,particles[gtid].position.y,particles[gtid].position.z);
		//printf("accels %d: (%f,%f,%f)\n",gtid,accels[gtid].x,accels[gtid].y,accels[gtid].z);
	}
}


// This device method accumulate acceleration for the current particle for the
// interaction with the particles in a tile
__device__ float3f accumulateAccels(Particle currentParticle, float3f accel, int itLenght){
	extern __shared__ Particle sharedParticles[];
  //printf("Device:: shared -> (%f,%f,%f)\n",sharedParticles[31].position.x,sharedParticles[31].position.y,sharedParticles[31].position.z);
	// iterate through each particle in the current tile
  // need to consider the last tile...
  // If the N is not evenly divisible by block dim, last tile will not have full list of particle to iterate.....
  // Need to fix it here!!!!!!!!!!!!!??????????????
	for(int i = 0; i < itLenght; i++){
		accel = interaction(currentParticle, sharedParticles[i], accel);
	}
	return accel;
}



// This is the global kernel to initialize calculate acceleration
__global__ void computeBodyForce(int N , Particle* particles, float3f* accels){
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	// defining array for in shared memory
	extern __shared__ Particle sharedParticles[];
	if(gtid < N){
		// Iterate through each tile
		int i, tile;
		float3f accel = float3f(0.0f,0.0f,0.0f);
		Particle currentParticle = particles[gtid];
    //printf("Device:: P %d -> (%f,%f,%f)\n",gtid,currentParticle.position.x,currentParticle.position.y,currentParticle.position.z);
		for(i = 0,tile = 0; i < N; i += blockDim.x, tile++){
			// get the particle to store in shared memory
			int idx = tile * blockDim.x + threadIdx.x;
			sharedParticles[threadIdx.x] = particles[idx];
			// Synchronize thread before going for updating acceleration for the current tile
			__syncthreads();

      // calculating the lenght of the shared memory to iterate through
      int itLenght = blockDim.x;
      if(tile == (int) N/blockDim.x){
        itLenght = N - tile * blockDim.x;
      }
      //printf("Lenght: %d\n", itLenght);
			// update acceleration for the current particle using the current tile
			accel = accumulateAccels(currentParticle, accel, itLenght);
			__syncthreads();
		}
		accels[gtid] = accel;
		//printf("p%d,%f %f %f\n",gtid,accels[gtid].x,accels[gtid].y,accels[gtid].z);
	}
}


// This file os for writing particle position in each interaction
std::ofstream output;

// This method start creating and executing the kernel
void startComputation(Particle* h_particles, int N){

  // opening file
  output.open("result.txt",std::ios_base::app);

  // checking whether the device allow us for overlap between copying and mkernel execution
  if(!ovelapCapabled()){
    exit(-1);
  }

  printf("Simulation started in device...\n");
  // Allocate buffer in device
	Particle* dev_particles;
	float3f* dev_accels;
	CUDA_CHECK_RETURN(cudaMalloc( (void**) &dev_particles, N * sizeof(Particle) ));
	CUDA_CHECK_RETURN(cudaMalloc( (void**) &dev_accels, N * sizeof(float3f) ));

	// initialize stream
	cudaStream_t stream; // need to think whether to use one or two streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream));

	// Asynchronous copying from host to device
	CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_particles, h_particles, N * sizeof(Particle),
			cudaMemcpyHostToDevice, stream));

	// define threads grid size  and block size
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

  // size fo the shared memory in each thread block
	unsigned int sharedMemorySize = blockDim.x * sizeof(Particle);

	// iteration loop
	for(int it = 0; it < NUM_IT; it++){
    // sampleKernel <<< 1, 32 >>> ();
		// kernel for compute body force
		computeBodyForce<<< gridDim , blockDim , sharedMemorySize , stream >>> (N,dev_particles,dev_accels);

		// kernel for position update
		updatePosition<<< gridDim , blockDim , 0 , stream >>> (N,dev_particles,dev_accels);

		// copy the position of the particles of the current step
    // Asynchronous copying from host to device
  	CUDA_CHECK_RETURN(cudaMemcpyAsync(h_particles, dev_particles, N * sizeof(Particle),
  			cudaMemcpyDeviceToHost, stream));

    cudaDeviceSynchronize();

    // write to file
    writeToFile(N,h_particles);
    //printf("It : %d\n",it);
	}

	// Synchsonize device with host
	CUDA_CHECK_RETURN(cudaStreamSynchronize( stream ));

	// Free the memory allocated in gpu
	CUDA_CHECK_RETURN(cudaFree(dev_particles));

  printf("Simulation finished in device...\n");
}

void writeToFile(int N, Particle* h_particles){
  for(int i = 0; i < N ; i++){
    output<<h_particles[i].position.x<<" "<<h_particles[i].position.y<<"  "
    <<h_particles[i].position.z<<std::endl;
  }
}

/**
 * This method check whether the device allow opvelap between copying and kernel execution
 */
bool ovelapCapabled(){
	cudaDeviceProp prop;
	int device;
	CUDA_CHECK_RETURN(cudaGetDevice(&device));
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device));

	if(!prop.deviceOverlap){
		std::cout<<"Device does not allow overlap ..."<<std::endl;
		return false;
	}

	return true;
}

/**
* This method checks whether two particles are approaching to each other
*/
__device__ bool isApproaching(float3f r, float3f currentVel, float3f otherVel){
  float3f relVel;
  relVel.x = otherVel.x - currentVel.x;
  relVel.y = otherVel.y - currentVel.y;
  relVel.z = otherVel.z - currentVel.z;

  float dotValue = r.x * relVel.x + r.y * relVel.y + r.z * relVel.z;
  if(dotValue > 0.0f){
    return false;
  } else{
    return true;
  }
}


/**
 * This method calculate interaction between two elements
 */
__device__ float3f interaction(Particle currentParticle, Particle otherParticle, float3f accel){
  // calculate the distance vector between two particles
  float3f r;
  r.x = otherParticle.position.x - currentParticle.position.x;
  r.y = otherParticle.position.y - currentParticle.position.y;
  r.z = otherParticle.position.z - currentParticle.position.z;

  float dist2 = r.x * r.x + r.y * r.y + r.z * r.z;
  // if r < epsilon , set r equal to epsilon and calculate accordingly
  if(dist2 < g_epsilon2) dist2 = g_epsilon2;

  float dist = sqrtf(dist2);
  // normalizing the distance vector
  r.x /= dist;
  r.y /= dist;
  r.z /= dist;

  // Calculate force if dist is greater than and equal to epsilon
  float otherMass = (otherParticle.pType == ParticleType::IRON) ? g_mass_fe : g_mass_si;
  float scale = 0.0f;

  if(g_diameter <= dist){
    // Case I
    scale = G * otherMass / dist2;
  } else{
    // Case II III IV
    if(currentParticle.pType == ParticleType :: IRON && otherParticle.pType == ParticleType :: IRON){
      // Both particles are of type IRON. No case III

      if(dist >= g_diameter - g_diameter * g_sh_depth_fe && dist < g_diameter){
        // Case II
        scale = G * otherMass / dist2 - g_k_fe * (g_diameter2 - dist2);
      } else if(dist >= g_epsilon && dist < g_diameter - g_diameter * g_sh_depth_fe){
        // Case IV
        // First checking whether the deparatin are increasing or decreasing
        bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
        if(approaching){
            //printf("dist2: %f\n",dist2);
            scale = G * otherMass / dist2 - g_k_fe * (g_diameter2 - dist2);
        } else{
            scale = G * otherMass / dist2 - g_k_fe * g_reduce_k_fe * (g_diameter2 - dist2);
        }
      }

    } else if(currentParticle.pType == ParticleType :: SILICA && otherParticle.pType == ParticleType :: SILICA){
      // both particles are of type SILICA. No case III

      if(dist >= g_diameter - g_diameter * g_sh_depth_si && dist < g_diameter){
        // Case II
        scale = G * otherMass / dist2 - g_k_si * (g_diameter2 - dist2);
      } else if(dist >= g_epsilon && dist < g_diameter - g_diameter * g_sh_depth_si){
        // Case IV
        // First checking whether the deparatin are increasing or decreasing
        bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
        if(approaching){
          scale = G * otherMass / dist2 - g_k_si * (g_diameter2 - dist2);
        } else{
          scale = G * otherMass / dist2 - g_k_si * g_reduce_k_si * (g_diameter2 - dist2);
        }
      }

    } else{
      // particles are of different types
      // Case II does not require to check approaching test
      if(dist >= g_diameter - g_diameter * g_sh_depth_si && dist < g_diameter){
        // Case II
        scale = G * otherMass / dist2 - 0.5f * (g_k_si + g_k_fe) * (g_diameter2 - dist2);
      } else{
        // Case III , IV
        // First checking whether the deparatin are increasing or decreasing
        bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
        if(dist >= g_diameter - g_diameter * g_sh_depth_fe && dist < g_diameter - g_diameter * g_sh_depth_si){
          // Case III
          if(approaching){
              scale = G * otherMass / dist2 - 0.5f * (g_k_si + g_k_fe) * (g_diameter2 - dist2);
          } else{
            scale = G * otherMass / dist2 - 0.5f * (g_k_si * g_reduce_k_si + g_k_fe) * (g_diameter2 - dist2);
          }
        } else if (dist >= g_epsilon && dist < g_diameter - g_diameter * g_sh_depth_fe){
          // Case IV
          if(approaching){
            scale = G * otherMass / dist2 - 0.5f * (g_k_si + g_k_fe) * (g_diameter2 - dist2);
          } else{
            scale = G * otherMass / dist2 - 0.5f * (g_k_si * g_reduce_k_si + g_k_fe * g_reduce_k_fe) * (g_diameter2 - dist2);
          }
        }

      }
    }
  }
  //printf("dist : %f\n", dist);
  // update acceleration
  accel.x += r.x * scale;
  accel.y += r.y * scale;
  accel.z += r.z * scale;
  //printf("accel : %f\n", accel.x);

	return accel;

}
