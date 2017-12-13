# Compile and run the program
#NVCC = nvcc -arch=sm_30 -std=c++11
NVCC = /usr/local/cuda-9.0/bin/nvcc -arch=sm_30 -std=c++11
GCC = g++ -std=c++11 -lpthread
INCDIR = -I./ -I./simulator -I./device
LIBDIR = -L./
#SPLIB = -L/pdc/vol/cuda/cuda-8.0/lib64
SPLIB = -L/usr/local/cuda-9.0/lib64
LIBS    = -lcudart -lpthread
FLAGS = $(INCDIR)


all: device.o main.out

main.out:  simulator.o
	$(GCC) $(FLAGS) main.cpp -o main.out *.o $(SPLIB) $(LIBS)
device.o:
	$(NVCC) $(FLAGS) -c device/device.cu -o device.o
simulator.o:
	$(NVCC) $(FLAGS) -c simulator/simulator.cu -o simulator.o
run:
	#srun ./main.out
	./main.out
setup:
	module load gcc/4.9.2 cuda/8.0
clean:
	rm *.out *.o *.txt
