TARGET_ARCH = linux64

include ../Makefile.config

PTHREADS := 1

CXXFLAGS += -std=c++11

PROJECT := kpn_neuralnet
SRCS    := $(wildcard *.cpp)
OBJS    := $(SRCS:.cpp=.o) $(filter-out ../../darknet/obj/darknet.o, $(wildcard ../../darknet/obj/*.o))

INCDIR  := -I../../darknet/include/ 

include ../Makefile.rules
