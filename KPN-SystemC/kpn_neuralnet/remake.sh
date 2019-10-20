#!/bin/bash

make clean
cd ../../darknet/
make 
cd ../KPN-SystemC/kpn_neuralnet/
make

