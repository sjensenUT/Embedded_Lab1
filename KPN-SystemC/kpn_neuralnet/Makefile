TARGET_ARCH = linux64

include ../Makefile.config

PTHREADS := 1

CXXFLAGS += -std=c++11

PROJECT := kpn_neuralnet
SRCS    := $(wildcard *.cpp)
OBJS    := $(SRCS:.cpp=.o) $(filter-out ../../darknet/obj/darknet.o, $(wildcard ../../darknet/obj/*.o))

INCDIR  := -I../../darknet/include/ 

include ../Makefile.rules

.PHONY: part1

part1:
	./kpn_neuralnet.x part1


.PHONY: part2a

part2a:
	./kpn_neuralnet.x part2a

.PHONY: part2b

part2b:
	./kpn_neuralnet.x part2b


