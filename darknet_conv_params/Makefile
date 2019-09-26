DARKNET = ../darknet

all:	extract load


load:
	gcc -c load.c -I $(DARKNET)/include/ -I $(DARKNET)/src/ -std=c99
	gcc	load.o\
		$(DARKNET)/obj/*_layer.o\
		$(DARKNET)/obj/box.o\
		$(DARKNET)/obj/blas.o\
		$(DARKNET)/obj/cuda.o\
		$(DARKNET)/obj/data.o\
		$(DARKNET)/obj/gemm.o\
		$(DARKNET)/obj/list.o\
		$(DARKNET)/obj/tree.o\
		$(DARKNET)/obj/utils.o\
		$(DARKNET)/obj/image.o\
		$(DARKNET)/obj/layer.o\
		$(DARKNET)/obj/matrix.o\
		$(DARKNET)/obj/col2im.o\
		$(DARKNET)/obj/im2col.o\
		$(DARKNET)/obj/parser.o\
		$(DARKNET)/obj/network.o\
		$(DARKNET)/obj/activations.o\
		$(DARKNET)/obj/option_list.o\
		-lm -lpthread -o load

extract:
	gcc -c extract.c -I $(DARKNET)/include/ -I $(DARKNET)/src/ -std=c99
	gcc	extract.o\
		$(DARKNET)/obj/*_layer.o\
		$(DARKNET)/obj/box.o\
		$(DARKNET)/obj/blas.o\
		$(DARKNET)/obj/cuda.o\
		$(DARKNET)/obj/data.o\
		$(DARKNET)/obj/gemm.o\
		$(DARKNET)/obj/list.o\
		$(DARKNET)/obj/tree.o\
		$(DARKNET)/obj/utils.o\
		$(DARKNET)/obj/image.o\
		$(DARKNET)/obj/layer.o\
		$(DARKNET)/obj/matrix.o\
		$(DARKNET)/obj/col2im.o\
		$(DARKNET)/obj/im2col.o\
		$(DARKNET)/obj/parser.o\
		$(DARKNET)/obj/network.o\
		$(DARKNET)/obj/activations.o\
		$(DARKNET)/obj/option_list.o\
		-lm -lpthread -o extract

clean:
	rm *.o load extract
