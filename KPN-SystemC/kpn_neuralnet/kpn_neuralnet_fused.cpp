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
using std::string;

typedef std::vector<std::string> strs;

const int BIGGEST_FIFO_SIZE = 416 * 416 * 16;
const int inputWidths[15] = {416, 416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13}; 
const int inputHeights[15] = {416, 416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13};
const int strides[15] = {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1};
const int filterSizes[15] = {3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3};
const string types[15] = {"convolutional", "maxpool", "convolutional", "maxpool", "convolutional", "maxpool", "convolutional", "maxpool"
                          "convolutional", "maxpool", "convolutional", "maxpool", "convolutional", "convolutional", "convolutional"};
const float ANCHORS[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                           5.47434, 7.88282 , 3.52778, 9.77052, 9.16828};

kpn_fused_thread::kpn_fused_thread(sc_module_name name, int coords[][9][4], int tileIndex) : sc_module(name)
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
    
    int tileCoords[16][4];
    int w[16];
    int h[16];
    for(int i = 0; i < 16; i++){
        for(int j = 0; j < 4; j++){
            tileCoords[i][j] = coords[i][tileIndex][j];
        }
        w[i] = tileCoords[i][2] - tileCoords[i][0] + 1;
        h[i] = tileCoords[i][3] - tileCoords[i][1] + 1;
    }

    conv14 = new conv_layer("conv14",14, w[14], h[14], 512, 1, 1, 425, 1, LINEAR, false, true, tileCoords[14], tileCoords[15], -1);
    conv14->in(*conv13_to_conv14);
    
    conv13 = new conv_layer("conv13",13, w[13], h[13], 1024, 3, 1, 512, 1, LEAKY, true, true, tileCoords[13], tileCoords[14], -1);
    conv13->in(*conv12_to_conv13);
    conv13->out(*conv13_to_conv14);
    
    conv12 = new conv_layer("conv12",12, w[12], h[12], 512, 3, 1, 1024, 1, LEAKY, true, true, tileCoords[12], tileCoords[13], -1);
    conv12->in(*max11_to_conv12);
    conv12->out(*conv12_to_conv13);
    
    max11 = new max_layer("max11",11, w[11], h[11], 512, 2, 1, true, tileCoords[11], tileCoords[12], -1);
    max11->in(*conv10_to_max11);
    max11->out(*max11_to_conv12);
    
    conv10 = new conv_layer("conv10",10, w[10], h[10], 256, 3, 1, 512, 1, LEAKY, true, true, tileCoords[10], tileCoords[11], -1);
    conv10->in(*max9_to_conv10);
    conv10->out(*conv10_to_max11);
    
    max9 = new max_layer("max9",9, w[9], h[9], 256, 2, 2, true, tileCoords[9], tileCoords[10], -1);
    max9->in(*conv8_to_max9);
    max9->out(*max9_to_conv10);

    conv8 = new conv_layer("conv8",8, w[8], h[8], 128, 3, 1, 256, 1, LEAKY ,true, true, tileCoords[8], tileCoords[9], -1);
    conv8->in(*max7_to_conv8);
    conv8->out(*conv8_to_max9);
    
    max7 = new max_layer("max7",7, w[7], h[7], 128, 2, 2, true, tileCoords[7], tileCoords[8], -1);
    max7->in(*conv6_to_max7);
    max7->out(*max7_to_conv8);
    
    conv6 = new conv_layer("conv6",6, w[6], h[6], 64, 3, 1, 128, 1, LEAKY, true, true, tileCoords[6], tileCoords[7], -1);
    conv6->in(*max5_to_conv6);
    conv6->out(*conv6_to_max7);

    max5 = new max_layer("max5",5, w[5], h[5], 64, 2, 2, true, tileCoords[5], tileCoords[6], -1);
    max5->in(*conv4_to_max5);
    max5->out(*max5_to_conv6);

    conv4 = new conv_layer("conv4",4, w[4], h[4], 32, 3, 1, 64, 1, LEAKY, true, true, tileCoords[4], tileCoords[5], -1);
    conv4->in(*max3_to_conv4);
    conv4->out(*conv4_to_max5);
    
    max3 = new max_layer("max3",3, w[3], h[3], 32, 2, 2, true, tileCoords[3], tileCoords[4], -1);
    max3->in(*conv2_to_max3);
    max3->out(*max3_to_conv4);
    
    conv2 = new conv_layer("conv2",2, w[2], h[2], 16, 3, 1, 32, 1, LEAKY, true, true, tileCoords[2], tileCoords[3], -1);
    conv2->in(*max1_to_conv2);
    conv2->out(*conv2_to_max3);
    
    max1 = new max_layer("max1",1, w[1], h[1], 16, 2, 2, true, tileCoords[1], tileCoords[2], -1);
    max1->in(*conv0_to_max1);
    max1->out(*max1_to_conv2);
    
    conv0 = new conv_layer("conv0",0, w[0], h[0], 3, 3,1,16, 1, LEAKY, true, true, tileCoords[0], tileCoords[1], -1);
    conv0->out(*conv0_to_max1);
}


kpn_neuralnet_fused::kpn_neuralnet_fused(sc_module_name name) : sc_module(name)
{
    cout << "instantiating fused kpn_neuralnet" << endl;
    //paths to example images
    strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
    
    //defining necessary fifo queues
    reader_to_scatter = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    merge_to_region = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    reader_to_writer    = new sc_fifo<float>(800 * 600 * 3);
    int_reader_to_writer    = new sc_fifo<int>(1); // needed to send im.w and im.h
    int2_reader_to_writer   = new sc_fifo<int>(1);
    char_reader_to_writer   = new sc_fifo<string>(1);
    for(int i = 0; i < 9; i++){
        scatter_to_nn[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
        nn_to_merge[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
    }

    reader0 = new image_reader("image_reader",images, -1);
    reader0->out(*reader_to_scatter);
    reader0->im_out(*reader_to_writer);
    reader0->im_w_out(*int_reader_to_writer);
    reader0->im_h_out(*int2_reader_to_writer);
    reader0->im_name_out(*char_reader_to_writer);

    
    region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                   true, 1, 1, true, 0.6, true, 13, 13, 425, -1);
    region->in(*merge_to_region);
    region->im_in(*reader_to_writer);
    region->im_w_in(*int_reader_to_writer);
    region->im_h_in(*int2_reader_to_writer);
    region->im_name_in(*char_reader_to_writer);
    
    int tileCoords[16][9][4];
    getTileCoords(13, 13, tileCoords[15]);
    for(int i = 14; i >= 0; i--){
        calcPrevCoords(tileCoords[i+1], tileCoords[i], strides[i], filterSizes[i], inputWidths[i], inputHeights[i], types[i]);
    }
     
    int *widths = new int[3]  { tileCoords[15][0][2] - tileCoords[15][0][0] + 1,
                                tileCoords[15][1][2] - tileCoords[15][1][0] + 1,
                                tileCoords[15][2][2] - tileCoords[15][2][0] + 1 };
    int *heights = new int[3] { tileCoords[15][0][3] - tileCoords[15][0][1] + 1,
                                tileCoords[15][3][3] - tileCoords[15][3][1] + 1,
                                tileCoords[15][6][3] - tileCoords[15][6][1] + 1 };
    scatter = new scatter_layer("scatter", tileCoords[0], 416, 416, 3);
    scatter->in(*reader_to_scatter);
    merge = new merge_layer("merge", widths, heights, 425);
    merge->out(*merge_to_region);
    for(int i = 0; i < 9; i++){
        threads[i] = new kpn_fused_thread("neuralnet_thread", tileCoords, i);
        threads[i]->conv0->in(*scatter_to_nn[i]);
        threads[i]->conv14->out(*nn_to_merge[i]);
        scatter->out[i](*scatter_to_nn[i]);
        merge->in[i](*nn_to_merge[i]);
    }
}
