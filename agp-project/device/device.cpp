#include <iostream>
#include "device.hpp"

// Cuda error check
static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
* Implementation of ComputeDevice class
*/

/**
* Forward method decleration
*/
__device__ float3f interaction(Particle currentParticle, Particle otherParticle, ModelParameters* modelParameters, float3f accel);


/**
* Constructor
*/
ComputeDevice::ComputeDevice(){

}

/**
* Constructor
*/

ComputeDevice::ComputeDevice(cudaGraphicsResource* resource, Particle* h_particles, ModelParameters modelParameters, const int BLOCK_SIZE){
  // mapped the shared resource and ask the cuda runtime for a pointer to the mapped resource
  size_t size;
  CUDA_CHECK_RETURN(cudaGraphicsMapResources(1,&resource,NULL));
  CUDA_CHECK_RETURN(cudaGraphicsResourceGetMappedPointer((void**) &dev_particles,&size,resource));

  // Checking the size to ensure
  // std::cout<<size<<std::endl;
  // std::cout<<modelParameters.nParticles * sizeof(Particle)<<std::endl;

  // allocate memory in the device for ModelParameters and acceleration buffer
  CUDA_CHECK_RETURN(cudaMalloc( (void**) &accels, modelParameters.nParticles * sizeof(float3f)));
  CUDA_CHECK_RETURN(cudaMalloc( (void**) &dev_modelParameters, 1 * sizeof(ModelParameters)));

  // Copying ModelParameters data from host memory  to device
  CUDA_CHECK_RETURN(cudaMemcpy(dev_modelParameters, &modelParameters, 1 * sizeof(ModelParameters),cudaMemcpyHostToDevice));

  // Setting kernel grid and block size
  blockDim = dim3(BLOCK_SIZE);
	gridDim = dim3((modelParameters.nParticles + blockDim.x - 1) / blockDim.x);
  sharedMemorySize = blockDim.x * sizeof(Particle);

}


// This device method accumulate acceleration for the current particle for the
// interaction with the particles in a tile
__device__ float3f accumulateAccels(Particle currentParticle, ModelParameters* modelParameters,float3f accel){
	extern __shared__ Particle sharedParticles[];
  //printf("Device:: shared -> pos(%f,%f,%f)\n",sharedParticles[0].position.x,sharedParticles[0].position.y,sharedParticles[0].position.z);
	// iterate through each particle in the current tile

	for(int i = 0; i < blockDim.x; i++){
		accel = interaction(currentParticle, sharedParticles[i],modelParameters, accel);
	}
	return accel;
}

// This is the global kernel to initialize calculate acceleration
__global__ void computeBodyForce(Particle* particles, float3f* accels, ModelParameters* modelParameters){
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("N : %d, Device -> pos(%f,%f,%f)\n",modelParameters[0].nParticles,particles[0].position.x, particles[0].position.y, particles[0].position.z);
	// defining array for in shared memory
	extern __shared__ Particle sharedParticles[];
	if(gtid < modelParameters[0].nParticles){
		// Iterate through each tile
		int i, tile;
		float3f accel = float3f(0.0f,0.0f,0.0f);
		Particle currentParticle = particles[gtid];
    //printf("Device:: P %d -> (%f,%f,%f)\n",gtid,currentParticle.position.x,currentParticle.position.y,currentParticle.position.z);
		for(i = 0,tile = 0; i < modelParameters[0].nParticles; i += blockDim.x, tile++){
			// get the particle to store in shared memory
			int idx = tile * blockDim.x + threadIdx.x;
			sharedParticles[threadIdx.x] = particles[idx];
			// Synchronize thread before going for updating acceleration for the current tile
			__syncthreads();

      //printf("Lenght: %d\n", itLenght);
			// update acceleration for the current particle using the current tile
			accel = accumulateAccels(currentParticle,modelParameters, accel);
			__syncthreads();
		}
		accels[gtid] = accel;
		//printf("p%d,%f %f %f\n",gtid,accels[gtid].x,accels[gtid].y,accels[gtid].z);
	}

}


// gpu kernel for position update
__global__ void updatePosition(Particle* particles, float3f* accels, ModelParameters* modelParameters){
	unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gtid < modelParameters[0].nParticles){
		// Numerical methods: Euler method. Update later to Leaf-frog method
    float3f prev_vel = particles[gtid].velocity;

    // update velocity
    particles[gtid].velocity.x += accels[gtid].x * modelParameters[0].g_time_step;
    particles[gtid].velocity.y += accels[gtid].y * modelParameters[0].g_time_step;
    particles[gtid].velocity.z += accels[gtid].z * modelParameters[0].g_time_step;
    // update position
    // position.x += vel.x * g_time_step;
    // position.y += vel.y * g_time_step;
    // position.z += vel.z * g_time_step;
    // Forward Euler
    // particles[i].position.x += g_time_step * (0.5f * accels[i].x * g_time_step + prev_vel.x);
    // particles[i].position.y += g_time_step * (0.5f * accels[i].y * g_time_step + prev_vel.y);
    // particles[i].position.z += g_time_step * (0.5f * accels[i].z * g_time_step + prev_vel.z);

    // Leap-frog method
    particles[gtid].position.x += modelParameters[0].g_time_step * (0.5f * accels[gtid].x * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.x + particles[gtid].velocity.x ));
    particles[gtid].position.y += modelParameters[0].g_time_step * (0.5f * accels[gtid].y * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.y + particles[gtid].velocity.y ));
    particles[gtid].position.z += modelParameters[0].g_time_step * (0.5f * accels[gtid].z * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.z + particles[gtid].velocity.z ));

	}
}

// This method is the entry point of Kernel
void ComputeDevice::compute(){
  // Call the kernel to calculate the force action on each particle
  computeBodyForce<<< gridDim , blockDim , sharedMemorySize >>> (dev_particles,accels,dev_modelParameters);
  // kernel for position update
  updatePosition<<< gridDim , blockDim>>> (dev_particles,accels,dev_modelParameters);
  cudaDeviceSynchronize();
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
__device__ float3f interaction(Particle currentParticle, Particle otherParticle, ModelParameters* modelParameters, float3f accel){
    // calculate the distance vector between two particles
    float3f r;
    r.x = otherParticle.position.x - currentParticle.position.x;
    r.y = otherParticle.position.y - currentParticle.position.y;
    r.z = otherParticle.position.z - currentParticle.position.z;

    float dist2 = r.x * r.x + r.y * r.y + r.z * r.z;
    // if r < epsilon , set r equal to epsilon and calculate accordingly
    if(dist2 < modelParameters[0].g_epsilon2) dist2 = modelParameters[0].g_epsilon2;

    float dist = sqrtf(dist2);
    // normalizing the distance vector
    r.x /= dist;
    r.y /= dist;
    r.z /= dist;

    // Calculate force if dist is greater than and equal to epsilon
    float currentMass = (currentParticle.pType == ParticleType::IRON) ? modelParameters[0].g_mass_fe : modelParameters[0].g_mass_si;
    float otherMass = (otherParticle.pType == ParticleType::IRON) ? modelParameters[0].g_mass_fe : modelParameters[0].g_mass_si;
    float scale = 0.0f;

    if(modelParameters[0].g_diameter <= dist){
      // Case I
      scale = modelParameters[0].G * currentMass * otherMass / dist2;
    } else{
      // Case II III IV
      if(currentParticle.pType == ParticleType :: IRON && otherParticle.pType == ParticleType :: IRON){
        // Both particles are of type IRON. No case III

        if(dist >= modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_fe && dist < modelParameters[0].g_diameter){
          // Case II
          scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (modelParameters[0].g_k_fe * (modelParameters[0].g_diameter2 - dist2));
        } else if(dist >= modelParameters[0].g_epsilon && dist < modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_fe){
          // Case IV
          // First checking whether the deparatin are increasing or decreasing
          bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
          if(approaching){
              //printf("dist2: %f\n",dist2);
              scale = (modelParameters[0].G* currentMass  * otherMass) / dist2 - (modelParameters[0].g_k_fe * (modelParameters[0].g_diameter2 - dist2));
          } else{
              scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (modelParameters[0].g_k_fe * modelParameters[0].g_reduce_k_fe * (modelParameters[0].g_diameter2 - dist2));
          }
        }

      } else if(currentParticle.pType == ParticleType :: SILICA && otherParticle.pType == ParticleType :: SILICA){
        // both particles are of type SILICA. No case III

        if(dist >= modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_si && dist < modelParameters[0].g_diameter){
          // Case II
          scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (modelParameters[0].g_k_si * (modelParameters[0].g_diameter2 - dist2));
        } else if(dist >= modelParameters[0].g_epsilon && dist < modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_si){
          // Case IV
          // First checking whether the deparatin are increasing or decreasing
          bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
          if(approaching){
            scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (modelParameters[0].g_k_si * (modelParameters[0].g_diameter2 - dist2));
          } else{
            scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (modelParameters[0].g_k_si * modelParameters[0].g_reduce_k_si * (modelParameters[0].g_diameter2 - dist2));
          }
        }

      } else{
        // particles are of different types
        // Case II does not require to check approaching test
        if(dist >= modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_si && dist < modelParameters[0].g_diameter){
          // Case II
          scale = (modelParameters[0].G * currentMass * otherMass)/dist2 - (0.5f * (modelParameters[0].g_k_si + modelParameters[0].g_k_fe) * (modelParameters[0].g_diameter2 - dist2));

        } else{
          // Case III , IV
          // First checking whether the deparatin are increasing or decreasing
          bool approaching = isApproaching(r, currentParticle.velocity, otherParticle.velocity);
          if(dist >= modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_fe && dist < modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_si){
            // Case III
            if(approaching){
                scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - ( 0.5f * (modelParameters[0].g_k_si + modelParameters[0].g_k_fe) * (modelParameters[0].g_diameter2 - dist2));
            } else{
              scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (0.5f * (modelParameters[0].g_k_si * modelParameters[0].g_reduce_k_si + modelParameters[0].g_k_fe) * (modelParameters[0].g_diameter2 - dist2));
            }
          } else if (dist >= modelParameters[0].g_epsilon && dist < modelParameters[0].g_diameter - modelParameters[0].g_diameter * modelParameters[0].g_sh_depth_fe){
            // Case IV
            if(approaching){
              scale = (modelParameters[0].G* currentMass  * otherMass) / dist2 - (0.5f * (modelParameters[0].g_k_si + modelParameters[0].g_k_fe) * (modelParameters[0].g_diameter2 - dist2));
            } else{
              scale = (modelParameters[0].G * currentMass * otherMass) / dist2 - (0.5f * (modelParameters[0].g_k_si * modelParameters[0].g_reduce_k_si + modelParameters[0].g_k_fe * modelParameters[0].g_reduce_k_fe) * (modelParameters[0].g_diameter2 - dist2));
            }
          }

        }
      }
    }
    //printf("dist : %f\n", dist);
    // update acceleration
    scale /= currentMass;
    accel.x += r.x * scale;
    accel.y += r.y * scale;
    accel.z += r.z * scale;
    //printf("accel : %f\n", accel.x);

  	return accel;

}


void ComputeDevice::releaseMemory(){
  cudaFree(dev_modelParameters);
  cudaFree(dev_particles);
  cudaFree(accels);
}

// Cuda error check
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err){
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

ComputeDevice::~ComputeDevice(){

}
