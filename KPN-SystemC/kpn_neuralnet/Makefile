TARGET_ARCH = linux64

include ../Makefile.config

PTHREADS := 1

CXXFLAGS += -std=c++11

PROJECT := kpn_neuralnet
SRCS    := $(wildcard *.cpp)
OBJS    := $(SRCS:.cpp=.o) ../../darknet/obj/*.o

include ../Makefile.rules
