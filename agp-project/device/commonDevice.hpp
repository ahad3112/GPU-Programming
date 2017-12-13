#ifndef COMMONDEVICE_HPP
#define COMMONDEVICE_HPP


/**
* Global variables related to device execution
*/
unsigned int BLOCK_SIZE	 = 32;								          // no. of threads in a block


/**
 *	Global variables
 *	(Supplementary Table 1| Parameters for Planar single-tailed collision)
 */

 unsigned int IT	 = 10000;									              // total number of iteration

/**
 *	Global variables
 *	(Supplementary Table 1| Parameters for Planar single-tailed collision)
 */

__device__ float G = 6.674e-20f; 							          // Universal Gravitational constant: unit [m3⋅kg−1⋅s−2.]
__device__ float g_diameter 	= 376.7800f;				      // diameter of an element
__device__ float g_diameter2 	= 376.7800f * 376.7800f;	// diameter square of an element
__device__ float g_mass_si		= 7.4161e+19f;
__device__ float g_mass_fe		= 1.9549e+20f;
__device__ float g_k_si		= 2.9114e+14f ;					      // repulsive parameter for silica
__device__ float g_k_fe		= 5.8228e+14f;					      // repulsive parameter for iron
__device__ float g_reduce_k_si	= 0.01f;					      // persent of reduction of the silicate repulsive force
__device__ float g_reduce_k_fe	= 0.02f;					      // persent of reduction of the iron repulsive force
__device__ float g_sh_depth_si = 0.001f; 					      // shell depth percent of silica
__device__ float g_sh_depth_fe = 0.002f;					      // shell depth percent of iron
__device__ float g_epsilon		= 47.0975f;					      // epsilon to avoid singularity
__device__ float g_epsilon2		= 47.0975f * 47.0975f;		// square of epsilon to avoid singularity
__device__ float g_time_step	= 5.8117f;					      // time step


#endif
