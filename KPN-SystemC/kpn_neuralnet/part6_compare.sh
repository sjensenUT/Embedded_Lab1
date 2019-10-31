#!/bin/bash



./kpn_neuralnet.x part5 > console_output5_1.txt

./kpn_neuralnet.x part6 > console_output6_1.txt

vim -c "vsp console_output5_1.txt" console_output6_1.txt
