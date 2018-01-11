
#include <iostream>
using namespace std;

#include "simulator.hpp"

// Global variable for simulation
// No. of particles . model case and No. of Iteration
const int N = 9600;
const int MODEL_CASE = 1; // Available model case: 1,2,3
const int MAX_IT = 1000;

// defining the window size
const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;

// thread block size
const int BLOCK_SIZE = 128;

// This is the entry point of the Earth Moon System simulation
int main(int argc, char ** argv){
  nbody::startSimulation(N, MODEL_CASE, MAX_IT, WINDOW_WIDTH, WINDOW_HEIGHT, BLOCK_SIZE);
  return 0;
}
