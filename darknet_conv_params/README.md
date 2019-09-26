This repository provides an application for extracting the convolutional layer parameters from darknet's weight files as well as an example of how to load these parameters.

# How to Extract Convolutional Parameters
_extract.c_ loads network configuration and weights of yolov2-tiny then stores the per-layer parameters in _out_ folder. To build and run this application:
1) Install darknet following the instructions on [their website](https://pjreddie.com/darknet/yolov2/).
2) Set _DARKNET_ in Makefile such that it points to the root of darknet installation.
3) _make_
4) _./extract_
5) Find parameters under _out/lx_ where _x_ is layer index.

# How to Load Extracted Parameters
_load.c_ creates a network corresponding to yolov2-tiny then loads the parameters for convolutional layers from files in _out_ folder. To build and run this application:
1) Install darknet following the instructions on [their website](https://pjreddie.com/darknet/yolov2/).
2) Set _DARKNET_ in Makefile such that it points to the root of darknet installation.
3) _make_
4) _./load_
