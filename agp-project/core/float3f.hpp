#ifndef FLOAT3F_HPP
#define FLOAT3F_HPP
class float3f{
  public:
  float x;
  float y;
  float z;
  __host__ __device__ float3f(float x, float y, float z);

  __host__ __device__ float3f();

  __host__ __device__ ~float3f();
};

#endif
