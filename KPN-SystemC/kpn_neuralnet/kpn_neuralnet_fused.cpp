#include <vector>
#include <string>
#include <iostream>
#include <systemc.h>
#include <stdio.h>
#include <stdlib.h>
#include "../kahn_process.h"

#include "darknet.h"
#include "array_ops.h"
#include "merge_scatter.h"
#include "../../darknet/src/convolutional_layer.h"
#include "../../darknet/src/maxpool_layer.h"
#include "../../darknet/src/region_layer.h"
#include "../../darknet/src/parser.h"
#include "../../darknet/src/activations.h"
#include "../../darknet/src/image.h"
#include "kpn_neuralnet.h"
#include "kpn_neuralnet_fused.h"
const int BIGGEST_FIFO_SIZE = 416 * 416 * 16;



//const int layerWidths[15] = {416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13, 13}; 
//const int layerHeights[15] = {416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13, 13};  
using std::string;


kpn_fused_thread::kpn_fused_thread(sc_module_name name, int *finalOutputCoords) : sc_module(name)
{
    conv0_to_max1   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max1_to_conv2   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv2_to_max3   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max3_to_conv4   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv4_to_max5   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max5_to_conv6   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv6_to_max7   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max7_to_conv8   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv8_to_max9   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max9_to_conv10   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv10_to_max11   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max11_to_conv12   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv12_to_conv13   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv13_to_conv14   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    
    //name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
    
    /*conv_layer::conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
     int _stride, int _numFilters, int _pad, ACTIVATION _activation,
     bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords)
    max_layer::max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
    int _stride, bool _crop, int* _inputCoords, int* _outputCoords)*/
    //void prevLayerCoords(int coords[4], int prevCoords[4], int stride, int filterSize, int prevWidth, int prevHeight, std::string layerType);
    
    int outputCoords[14][4];
    int initialInputCoords[4];
    int w[15];
    int h[15];
    prevLayerCoords(finalOutputCoords, outputCoords[13], 1, 3, 13, 13, "convolutional");
    w[14] = outputCoords[13][2] - outputCoords[13][0] + 1;
    h[14] = outputCoords[13][3] - outputCoords[13][1] + 1; 
    conv14 = new conv_layer("conv14",14, w[14], w[14], 512, 1, 1, 425, 1, LINEAR, false, true, outputCoords[13], finalOutputCoords);
    conv14->in(*conv13_to_conv14);
    
    prevLayerCoords(outputCoords[13], outputCoords[12], 1, 3, 13, 13, "convolutional");
    w[13] = outputCoords[12][2] - outputCoords[12][0] + 1;
    h[13] = outputCoords[12][3] - outputCoords[12][1] + 1;
    conv13 = new conv_layer("conv13",13, w[13], h[13], 1024, 3, 1, 512, 1, LEAKY, true, true, outputCoords[12], outputCoords[13]);
    conv13->in(*conv12_to_conv13);
    conv13->out(*conv13_to_conv14);
    
    prevLayerCoords(outputCoords[12], outputCoords[11], 1, 3, 13, 13, "convolutional");
    w[12] = outputCoords[11][2] - outputCoords[11][0] + 1;
    h[12] = outputCoords[11][3] - outputCoords[11][1] + 1;
    conv12 = new conv_layer("conv12",12, w[12], h[12], 512, 3, 1, 1024, 1, LEAKY, true, true, outputCoords[11], outputCoords[12]);
    conv12->in(*max11_to_conv12);
    conv12->out(*conv12_to_conv13);
    
    // !!! NOTE !!! this is the only max layer with stride=1
    prevLayerCoords(outputCoords[11], outputCoords[10], 1, 2, 13, 13, "maxpool");
    w[11] = outputCoords[10][2] - outputCoords[10][0] + 1;
    h[11] = outputCoords[10][3] - outputCoords[10][1] + 1;
    max11 = new max_layer("max11",11, w[11], h[11], 512, 2, 1, true, outputCoords[10], outputCoords[12]);
    max11->in(*conv10_to_max11);
    max11->out(*max11_to_conv12);
    
    prevLayerCoords(outputCoords[10], outputCoords[9], 1, 3, 26, 26, "convolutional");
    w[10] = outputCoords[9][2] - outputCoords[9][0] + 1;
    h[10] = outputCoords[9][3] - outputCoords[9][1] + 1;
    conv10 = new conv_layer("conv10",10, w[10], h[10], 256, 3, 1, 512, 1, LEAKY, true, true, outputCoords[9], outputCoords[10]);
    conv10->in(*max9_to_conv10);
    conv10->out(*conv10_to_max11);
    
    prevLayerCoords(outputCoords[9], outputCoords[8], 2, 2, 26, 26, "maxpool");
    w[9] = outputCoords[8][2] - outputCoords[8][0] + 1;
    h[9] = outputCoords[8][3] - outputCoords[8][1] + 1;
    max9 = new max_layer("max9",9, w[9], h[9], 256, 2, 2, true, outputCoords[8], outputCoords[9]);
    max9->in(*conv8_to_max9);
    max9->out(*max9_to_conv10);

    prevLayerCoords(outputCoords[8], outputCoords[7], 1, 3, 52, 52, "convolutional");
    w[8] = outputCoords[7][2] - outputCoords[7][0] + 1;
    h[8] = outputCoords[7][3] - outputCoords[7][1] + 1;
    conv8 = new conv_layer("conv8",8, w[8], h[8], 128, 3, 1, 256, 1, LEAKY ,true, true, outputCoords[7], outputCoords[8]);
    conv8->in(*max7_to_conv8);
    conv8->out(*conv8_to_max9);
    
    
    prevLayerCoords(outputCoords[7], outputCoords[6], 2, 2, 52, 52, "maxpool");
    w[7] = outputCoords[6][2] - outputCoords[6][0] + 1;
    h[7] = outputCoords[6][3] - outputCoords[6][1] + 1;
    max7 = new max_layer("max7",7, w[7], h[7], 128, 2, 2, true, outputCoords[6], outputCoords[7]);
    max7->in(*conv6_to_max7);
    max7->out(*max7_to_conv8);
    
    prevLayerCoords(outputCoords[6], outputCoords[5], 1, 3, 104, 104, "convolutional");
    w[6] = outputCoords[5][2] - outputCoords[5][0] + 1;
    h[6] = outputCoords[5][3] - outputCoords[5][1] + 1;
    conv6 = new conv_layer("conv6",6, w[6], h[6], 64, 3, 1, 128, 1, LEAKY, true, true, outputCoords[5], outputCoords[6]);
    conv6->in(*max5_to_conv6);
    conv6->out(*conv6_to_max7);

    prevLayerCoords(outputCoords[5], outputCoords[4], 2, 2, 104, 104, "maxpool");
    w[5] = outputCoords[4][2] - outputCoords[4][0] + 1;
    h[5] = outputCoords[4][3] - outputCoords[4][1] + 1;
    max5 = new max_layer("max5",5, w[5], h[5], 64, 2, 2, true, outputCoords[4], outputCoords[5]);
    max5->in(*conv4_to_max5);
    max5->out(*max5_to_conv6);

    prevLayerCoords(outputCoords[4], outputCoords[3], 1, 3, 208, 208, "convolutional");
    w[4] = outputCoords[3][2] - outputCoords[3][0] + 1;
    h[4] = outputCoords[3][3] - outputCoords[3][1] + 1;
    conv4 = new conv_layer("conv4",4, w[4], h[4], 32, 3, 1, 64, 1, LEAKY, true, true, outputCoords[3], outputCoords[4]);
    conv4->in(*max3_to_conv4);
    conv4->out(*conv4_to_max5);
    
    prevLayerCoords(outputCoords[3], outputCoords[2], 2, 2, 208, 208, "maxpool");
    w[3] = outputCoords[2][2] - outputCoords[2][0] + 1;
    h[3] = outputCoords[2][3] - outputCoords[2][1] + 1;
    max3 = new max_layer("max3",3, w[3], h[3], 32, 2, 2, true, outputCoords[2], outputCoords[3]);
    max3->in(*conv2_to_max3);
    max3->out(*max3_to_conv4);
    
    prevLayerCoords(outputCoords[2], outputCoords[1], 1, 3, 416, 416, "convolutional");
    w[2] = outputCoords[1][2] - outputCoords[1][0] + 1;
    h[2] = outputCoords[1][3] - outputCoords[1][1] + 1;
    conv2 = new conv_layer("conv2",2, w[2], h[2], 16, 3, 1, 32, 1, LEAKY, true, true, outputCoords[1], outputCoords[2]);
    conv2->in(*max1_to_conv2);
    conv2->out(*conv2_to_max3);
    
    prevLayerCoords(outputCoords[1], outputCoords[0], 2, 2, 416, 416, "maxpool");
    w[1] = outputCoords[0][2] - outputCoords[0][0] + 1;
    h[1] = outputCoords[0][3] - outputCoords[0][1] + 1;
    max1 = new max_layer("max1",1, w[1], h[1], 16, 2, 2, true, outputCoords[0], outputCoords[1]);
    max1->in(*conv0_to_max1);
    max1->out(*max1_to_conv2);
    
    prevLayerCoords(outputCoords[4], initialInputCoords, 1, 3, 416, 416, "convolutional");
    w[0] = initialInputCoords[2] - initialInputCoords[0] + 1;
    h[0] = initialInputCoords[3] - initialInputCoords[1] + 1;
    conv0 = new conv_layer("conv0",0, w[0], h[0], 3, 3,1,16, 1, LEAKY, true, true, initialInputCoords, outputCoords[0]);
    conv0->out(*conv0_to_max1);
}
