#!/bin/bash
#vim "../../darknet/console_output_darknet.txt"
cd ../../darknet/
make
./run_darknet.sh > console_output_darknet.txt
cd ../KPN-SystemC/kpn_neuralnet/
make
timeout 5 ./kpn_neuralnet.x > console_output.txt

vim -c "vsp console_output.txt" ../../darknet/console_output_darknet.txt
