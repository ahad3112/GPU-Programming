#include "cpuSim.hpp"

/*
* Implementation of simulation in CPU only
*/

// THis method update particles position in cpu
void updatePositionCPU(Particle* particles, float3f* accels, ModelParameters* modelParameters){
	for(int i = 0; i < modelParameters[0].nParticles; i++){
		// Numerical methods: Euler method. Update later to Leaf-frog method
    float3f prev_vel = particles[i].velocity;

    // update velocity
    particles[i].velocity.x += accels[i].x * modelParameters[0].g_time_step;
    particles[i].velocity.y += accels[i].y * modelParameters[0].g_time_step;
    particles[i].velocity.z += accels[i].z * modelParameters[0].g_time_step;
    // update position

    // Leap-frog method
    particles[i].position.x += modelParameters[0].g_time_step * (0.5f * accels[i].x * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.x + particles[i].velocity.x ));
    particles[i].position.y += modelParameters[0].g_time_step * (0.5f * accels[i].y * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.y + particles[i].velocity.y ));
    particles[i].position.z += modelParameters[0].g_time_step * (0.5f * accels[i].z * modelParameters[0].g_time_step +
      0.5f * ( prev_vel.z + particles[i].velocity.z ));

		//printf("p%d: (%f,%f,%f)\n",gtid,particles[gtid].position.x,particles[gtid].position.y,particles[gtid].position.z);
		//printf("accels %d: (%f,%f,%f)\n",gtid,accels[gtid].x,accels[gtid].y,accels[gtid].z);
	}
}

/**
* This method checks whether two particles are approaching to each other
*/
bool isApproachingCPU(float3f r, float3f currentVel, float3f otherVel){
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
float3f interactionCPU(Particle currentParticle, Particle otherParticle, ModelParameters* modelParameters, float3f accel){
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
          bool approaching = isApproachingCPU(r, currentParticle.velocity, otherParticle.velocity);
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
          bool approaching = isApproachingCPU(r, currentParticle.velocity, otherParticle.velocity);
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
          bool approaching = isApproachingCPU(r, currentParticle.velocity, otherParticle.velocity);
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


void updateEarthMoonSystemCPU(Particle* particles, ModelParameters* modelParameters){
  float3f* accels = (float3f*) malloc (modelParameters[0].nParticles * sizeof(float3f));
  for(int i = 0; i < modelParameters[0].nParticles; i++){
    accels[i] = float3f(0.0f,0.0f,0.0f);
    for(int j = 0; j < modelParameters[0].nParticles; j++){
      accels[i] = interactionCPU(particles[i], particles[j], modelParameters ,accels[i]);
    }
  }
  updatePositionCPU(particles, accels,modelParameters);
}
