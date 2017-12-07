#include<iostream>
#include "simulator.hpp"
#include "device.hpp"

void nbody::startSimulation(){
  std::cout<<"Simulation started..."<<std::endl;
  // Initialize the Particle list
  Particle p = Particle();

  // Call cuda for initialization and execution of the kernel
  startComputation();
}