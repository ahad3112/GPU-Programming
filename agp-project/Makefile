
NVCC = /usr/local/cuda-9.1/bin/nvcc -arch=sm_30 -std=c++11
CPP     = g++ -std=c++11 -Wall --pedantic
INCDIR  = -I./ -I./util -I./sphere -I./modules/build/include -I./simulator -I./renderer -I./device
LIBDIR  = -L./ -L./modules/build/lib
BINDIR  = ./bin
OBJDIR  = $(BINDIR)/obj
LIBS    = -lglad -lglfw3 -lGL -lEGL -lXrandr -lXext -lX11 -lrt -ldl -pthread -lcudart
CFLAGS  = $(INCDIR) $(LIBDIR)
SPLIB = -L/usr/local/cuda-9.1/lib64


MKDIR   = mkdir -p

all: setup main.out

main.out: device.o utilities.o util.o renderer.o simulator.o  sphere.o
	@$(CPP) $(CFLAGS) main.cpp -o $(BINDIR)/main.out $(OBJDIR)/*.o $(SPLIB) $(LIBS)

# The following line is an example on how to compile your own clases. Use
# the $(OBJDIR) directory to output your object files:
#
# example_class.o:
#	@$(CPP) $(CFLAGS) -c example_class.cpp \
#					  -o $(OBJDIR)/example_class.o
#
# Do not forget to specify example_class.o alongside the main.out label. This
# way, we tell make that main.out depends on example_class.o, triggering the
# build beforehand.
device.o:
	$(NVCC) $(CFLAGS) -c device/device.cu -o $(OBJDIR)/device.o

renderer.o:
	@$(NVCC) $(CFLAGS) -c renderer/renderer.cu -o $(OBJDIR)/renderer.o
simulator.o:
	$(NVCC) $(CFLAGS) -c simulator/simulator.cu -o $(OBJDIR)/simulator.o
sphere.o:
	@$(CPP) $(CFLAGS) -c sphere/sphere.cpp -o $(OBJDIR)/sphere.o

util.o:
	@$(NVCC) $(CFLAGS) -c util/util.cu -o $(OBJDIR)/util.o

utilities.o:
		$(NVCC) $(CFLAGS) -c util/utilities.cu -o $(OBJDIR)/utilities.o

setup:
	@$(MKDIR) $(BINDIR) $(OBJDIR)

clean:
	@$(RM) -rf $(BINDIR) $(OBJDIR)

rebuild: clean all
