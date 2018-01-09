#include "float3f.hpp"
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
