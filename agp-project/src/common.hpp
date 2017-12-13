/*
 * common.hpp
 *
 *  Created on: Nov 30, 2017
 *      Author: rohin
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <cmath>
#include "device.hpp"

 unsigned int NUM_PARTICLES 	= 100;							        // total no. of particles

 /**
 * Some constant mathematical variables
 */
 const float PI = 3.1415927;                             // value of PI

/*
float g_diameter 	= 376780.0f;				      // diameter of an element
float g_diameter2 	= 376780.0f * 376780.0f;	// diameter square of an element
float g_mass_si		= 7.4161e+19f;
float g_mass_fe		= 1.9549e+20f;
float g_k_si		= 2.9114e+11f;					      // repulsive parameter for silica
float g_k_fe		= 5.8228e+11f;					      // repulsive parameter for iron
float g_reduce_k_si	= 0.01f;					      // persent of reduction of the silicate repulsive force
float g_reduce_k_fe	= 0.02f;					      // persent of reduction of the iron repulsive force
float g_sh_depth_si = 0.001f; 					      // shell depth percent of silica
float g_sh_depth_fe = 0.002f;					      // shell depth percent of iron
float g_epsilon		= 47097.5f;					      // epsilon to avoid singularity
float g_time_step	= 5.8117f;					      // time step
*/

/**
 * Other constants
 * (Sourced from Internet)
 * */


float PERCENT_IRON = 0.30f;                             // percentage of iron in a collidor
float g_radius_earth  = 6371;
float g_radius_core_fe;


/*
float3f g_center_mass_one = float3f(23925.0f,0.0f,9042.7f);
float3f g_center_mass_two = float3f(-23925.0f,0.0f,-9042.7f);

float3f g_linear_velocity_one = float3f(-3.24160f,0.0f,0.0f);
float3f g_linear_velocity_two = float3f(3.24160f,0.0f,0.0f);

float3f g_angular_velocity_one = float3f(0.0f,8.6036e-4f,0.0f);
float3f g_angular_velocity_two = float3f(0.0f,-8.6036e-4f,0.0f);
*/
float3f g_center_mass_one = float3f(23925000.0f,0.0f,9042700.0f);
float3f g_center_mass_two = float3f(-23925000.0f,0.0f,-9042700.0f);

float3f g_linear_velocity_one = float3f(-3241.60f,0.0f,0.0f);
float3f g_linear_velocity_two = float3f(3241.60f,0.0f,0.0f);

float3f g_angular_velocity_one = float3f(0.0f,8.6036e-4f,0.0f);
float3f g_angular_velocity_two = float3f(0.0f,-8.6036e-4f,0.0f);

#endif /* COMMON_HPP_ */
