# Compile and run the program
NVCC = /usr/local/cuda-9.1/bin/nvcc -x cu -arch=sm_30 -std=c++11
GCC = g++ -std=c++11
INCDIR = -I./ -I./simulator -I./core -I./model -I./device -I./renderer -I./util -I./cpuSim -I./sphere -I./modules/build/include
LIBDIR  = -L./ -L./modules/build/lib
SPLIB = -L/pdc/vol/cuda/cuda-8.0/lib64
LIBS    = -lglad -lglfw3 -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -lcudart
FLAGS = $(INCDIR)

objects = main.o simulator.o model.o modelParameters.o device.o cpuSim.o renderer.o utilities.o util.o particle.o float3f.o

all: $(objects)
	/usr/local/cuda-9.1/bin/nvcc -arch=sm_30 $(LIBDIR) $(objects) -o main.out $(LIBS)
main.o:
	$(NVCC) $(INCDIR) -dc main.cpp -o main.o
simulator.o:
	$(NVCC) $(INCDIR) -dc simulator/simulator.cpp -o simulator.o
model.o:
	$(NVCC) $(INCDIR) -dc model/model.cpp -o model.o
modelParameters.o:
	$(NVCC) $(INCDIR) -dc model/modelParameters.cpp -o modelParameters.o
device.o:
	$(NVCC) $(INCDIR) -dc device/device.cpp -o device.o
cpuSim.o:
	$(NVCC) $(INCDIR) -dc cpuSim/cpuSim.cpp -o cpuSim.o
renderer.o:
	$(NVCC) $(INCDIR) -dc renderer/renderer.cpp -o renderer.o
utilities.o:
	$(NVCC) $(INCDIR) -dc util/utilities.cpp -o utilities.o
util.o:
	$(NVCC) $(INCDIR) -dc util/util.cpp -o util.o
particle.o:
	$(NVCC) $(INCDIR) -dc core/particle.cpp -o particle.o
float3f.o:
	$(NVCC) $(INCDIR) -dc core/float3f.cpp -o float3f.o
runf:
	srun nvprof ./main.out
run:
	srun ./main.out
setup:
	module load gcc/4.9.2 cuda/8.0
clean:
	rm -f *.o *.out *.txt
rebuild: clean all
