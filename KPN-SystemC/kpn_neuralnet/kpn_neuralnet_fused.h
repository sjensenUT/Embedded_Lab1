#ifndef KPN_NEURALNET_FUSED_H
#define KPN_NEURALNET_FUSED_H

#include<string>
#include "kpn_neuralnet.h"
using std::string;

class   kpn_fused_thread : public sc_module
{
    public:

    //Declare all queues between our layers here
    //I think the data type for all of them will be image
    sc_fifo<float> *conv0_to_max1,
            *max1_to_conv2,
            *conv2_to_max3,
            *max3_to_conv4,
            *conv4_to_max5,
            *max5_to_conv6,
            *conv6_to_max7,
            *max7_to_conv8,
            *conv8_to_max9,
            *max9_to_conv10,
            *conv10_to_max11,
            *max11_to_conv12,
            *conv12_to_conv13,
            *conv13_to_conv14;

    //Declare all layers here
    max_layer   *max1, *max3, *max5, *max7, *max9, *max11;
    conv_layer  *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;

    //Constructor of the overall network. Initialize all queues and layers
    kpn_fused_thread(sc_module_name name, int coords[][9][4], int tileIndex);
};

class   kpn_neuralnet_fused : public sc_module
{
    public:

    //Declare all queues between our layers here
    //I think the data type for all of them will be image
    sc_fifo<float>  *reader_to_scatter,
            *scatter_to_nn[9],
            *nn_to_merge[9],
            *merge_to_region;
    sc_fifo<float>  *reader_to_writer;
    sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer;
    sc_fifo<string>  *char_reader_to_writer;
    kpn_fused_thread *threads[9];
    region_layer *region;
    image_reader *reader0;
    scatter_layer *scatter;
    merge_layer *merge;
    kpn_neuralnet_fused(sc_module_name name);
};

#endif //KPN_NEURALNET_FUSED
