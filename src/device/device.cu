#include <iostream>
#include <stdio.h>
#include "device.hpp"
#include "commonDevice.hpp"


static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

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

// This method start creating and executing the kernel
void startComputation(Particle* h_particles, int N){

  // checking whether the device allow us for overlap between copying and mkernel execution
  if(!ovelapCapabled()){
    exit(-1);
  }

  printf("Simulation started in device...\n");
  // Allocate buffer in device
	Particle* dev_particles;
	float3* dev_accels;
	CUDA_CHECK_RETURN(cudaMalloc( (void**) &dev_particles, N * sizeof(Particle) ));
	CUDA_CHECK_RETURN(cudaMalloc( (void**) &dev_accels, N * sizeof(float3) ));

	// initialize stream
	cudaStream_t stream; // need to think whether to use one or two streams
	CUDA_CHECK_RETURN(cudaStreamCreate(&stream));

	// Asynchronous copying from host to device
	CUDA_CHECK_RETURN(cudaMemcpyAsync(dev_particles, h_particles, N * sizeof(Particle),
			cudaMemcpyHostToDevice, stream));

	// define threads grid size  and block size
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
	unsigned int sharedMemorySize = blockDim.x * sizeof(Particle);

	// iteration loop
	for(int it = 0; it < NUM_IT; it++){
    sampleKernel <<< 1, 32 >>> ();
		// kernel for compute body force
		//computeBodyForce<<< gridDim , blockDim , sharedMemorySize , stream >>> (NUM_PARTICLES,dev_particles,dev_accels);

		// kernel for position update
		//updatePosition<<< gridDim , blockDim , sharedMemorySize , stream >>> (NUM_PARTICLES,dev_particles,dev_accels);

		// copy the position of the particles of the current step if required
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
