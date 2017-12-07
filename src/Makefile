# Compile and run the program
NVCC = nvcc -arch=sm_30 -std=c++11
GCC = g++ -std=c++11
INCDIR = -I./ -I./simulator -I./particle
LIBDIR = -L./
LIBS    = -lglad -lglfw3 -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread
FLAGS = $(INCDIR)


all: device.o main.out

main.out: simulator.o
	$(GCC) $(FLAGS) main.cpp -o main.out *.o

device.o: particle.o
	$(NVCC) -c cuda/device.cu -o device.o

simulator.o:
	$(GCC) $(FLAGS) -c simulator/simulator.cpp -o simulator.o
particle.o:
	$(GCC) $(FLAGS) -c particle/particle.cpp -o particle.o
run:
	./main.out
setup:
	module load gcc/4.9.2 cuda/8.0
clean:
	rm *.out *.o
