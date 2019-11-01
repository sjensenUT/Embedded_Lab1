TARGET_ARCH = linux64

include ../Makefile.config

PTHREADS := 1

CXXFLAGS += -std=c++11

PROJECT := kpn_neuralnet
SRCS    := $(wildcard *.cpp) 
DEPS	:= $(wildcard *.h)
OBJS    := $(SRCS:.cpp=.o) $(filter-out ../../darknet/obj/darknet.o, $(wildcard ../../darknet/obj/*.o))

INCDIR  := -I../../darknet/include/ 

include ../Makefile.rules

.PHONY: part1

part1:
	./kpn_neuralnet.x part1

.PHONY: part2

part2:
	./kpn_neuralnet.x part2

.PHONY: part3
	
part3:
	./kpn_neuralnet.x part3

.PHONY: part5

part5:
	./kpn_neuralnet.x part5

.PHONY: part6

part6:
	./kpn_neuralnet.x part6
