#include <iostream>
#include <stdio.h>
#include "device.hpp"
#include "commonDevice.hpp"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

// defining particle and float3f
  __host__ __device__ float3f::float3f(float x, float y, float z) : x(x), y(y), z(z){

  }

  __host__ __device__ float3f::float3f(){
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
  }

  __host__ __device__ float3f::~float3f(){

  }

  __host__ __device__ Particle::Particle(float3f position, float3f velocity, ParticleType pType) :
  position(position), velocity(velocity), pType(pType)
  {
    //std::cout<<"Initializing particle..."<<std::endl;
  }

  __host__ __device__ Particle::~Particle(){

  }

/**
* Forward method decleration
*/
bool ovelapCapabled();

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


// This device method accumulate acceleration for the current particle for the
// interaction with the particles in a tile
__device__ float3f accumulateAccels(Particle currentParticle, float3f accel){
	extern __shared__ Particle sharedParticles[];
  printf("Device:: shared -> (%f,%f,%f)\n",sharedParticles[31].position.x,sharedParticles[31].position.y,sharedParticles[31].position.z);
	// iterate through each particle in the current tile
	for(int i = 0; i < blockDim.x; i++){
		// accel = interaction(currentParticle, sharedParticles[i], accel);
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
		for(i = 0,tile = 0; i <N; i += blockDim.x, tile++){
			// get the particle to store in shared memory
			int idx = tile * blockDim.x + threadIdx.x;
			sharedParticles[threadIdx.x] = particles[idx];
			// Synchronize thread before going for updating acceleration for the current tile
			__syncthreads();

			// update acceleration for the current particle using the current tile
			accel = accumulateAccels(currentParticle, accel);
			__syncthreads();
		}
		accels[gtid] = accel;
		//printf("p%d,%f %f %f\n",gtid,accels[gtid].x,accels[gtid].y,accels[gtid].z);
	}
}



// This method start creating and executing the kernel
void startComputation(Particle* h_particles, int N){

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
		//updatePosition<<< gridDim , blockDim , sharedMemorySize , stream >>> (NUM_PARTICLES,dev_particles,dev_accels);

		// copy the position of the particles of the current step
    // Asynchronous copying from host to device
  	CUDA_CHECK_RETURN(cudaMemcpyAsync(h_particles, dev_particles, N * sizeof(Particle),
  			cudaMemcpyDeviceToHost, stream));
	}

	// Synchsonize device with host
	CUDA_CHECK_RETURN(cudaStreamSynchronize( stream ));

	// Free the memory allocated in gpu
	CUDA_CHECK_RETURN(cudaFree(dev_particles));

  printf("Simulation finished in device...\n");
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
