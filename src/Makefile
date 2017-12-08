# Compile and run the program
NVCC = nvcc -arch=sm_30 -std=c++11
GCC = g++ -std=c++11
INCDIR = -I./ -I./simulator -I./particle -I./device -I./mathUtil
LIBDIR = -L./
SPLIB = -L/pdc/vol/cuda/cuda-8.0/lib64
LIBS    = -lcudart
FLAGS = $(INCDIR)


all: mathUtil.o particle.o device.o main.out

main.out:  simulator.o
	$(GCC) $(FLAGS) main.cpp -o main.out *.o $(SPLIB) $(LIBS)

mathUtil.o:
	$(GCC) $(FLAGS) -c mathUtil/mathUtil.cpp -o mathUtil.o

device.o:
	$(NVCC) $(FLAGS) -c device/device.cu -o device.o

simulator.o:
	$(GCC) $(FLAGS) -c simulator/simulator.cpp -o simulator.o
particle.o:
	$(GCC) $(FLAGS) -c particle/particle.cpp -o particle.o
run:
	srun ./main.out
setup:
	module load gcc/4.9.2 cuda/8.0
clean:
	rm *.out *.o
