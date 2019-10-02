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

//const int layerWidths[15] = {416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13, 13}; 
//const int layerHeights[15] = {416, 416, 208, 208, 104, 104, 52, 52, 26, 26, 13, 13, 13, 13, 13};  
using std::string;


class	kpn_fused_thread : public sc_module
{
	public:
	
    // Declare all queues between our layers here
    // I think the data type for all of them will be image
	sc_fifo<float>	*in_to_conv0, 
			*conv0_to_max1, 
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
			*conv13_to_conv14,
			*conv14_to_out;

	sc_fifo<float>  *reader_to_writer; 
	sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	sc_fifo<string>  *char_reader_to_writer; 
			//*conv2_to_detection,
			//*detection_to_writer;

    // Declare all layers here
	max_layer	*max3, *max5, *max7, *max9, *max11;
	conv_layer	*conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;
	conv_layer_unfused *conv0;
    max_layer_unfused *max1;
    region_layer	*region;
	image_reader	*reader0;
//	image_writer	*writer0;

  // Constructor of the overall network. Initialize all queues and layers
	kpn_fused_thread(sc_module_name name, int *finalOutputCoords) : sc_module(name)
	{
		/*strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
		reader_to_conv0 	= new sc_fifo<float*>(1);
		conv0_to_max1   	= new sc_fifo<float*>(1);
		max1_to_conv2   	= new sc_fifo<float*>(1);
		conv2_to_max3   	= new sc_fifo<float*>(1);
        max3_to_conv4   	= new sc_fifo<float*>(1);
		conv4_to_max5   	= new sc_fifo<float*>(1);
        max5_to_conv6   	= new sc_fifo<float*>(1);
		conv6_to_max7   	= new sc_fifo<float*>(1);
        max7_to_conv8   	= new sc_fifo<float*>(1);
		conv8_to_max9   	= new sc_fifo<float*>(1);
        max9_to_conv10   	= new sc_fifo<float*>(1);
		conv10_to_max11   	= new sc_fifo<float*>(1);
        max11_to_conv12   	= new sc_fifo<float*>(1);
		conv12_to_conv13   	= new sc_fifo<float*>(1);
        conv13_to_conv14   	= new sc_fifo<float*>(1);
		conv14_to_region   	= new sc_fifo<float*>(1);
		reader_to_writer 	= new sc_fifo<float*>(1); 
		int_reader_to_writer	= new sc_fifo<int>(1); // needed to send im.w and im.h
		int2_reader_to_writer 	= new sc_fifo<int>(1); 
		char_reader_to_writer  	= new sc_fifo<string>(1);
		
        // Here is where we will indicate the parameters for each layer. These can
        // be found in the cfg file for yolov2-tiny in the darknet folder.
	    reader0 = new image_reader("image_reader",images);
		reader0->out(*reader_to_conv0);
		reader0->im_out(*reader_to_writer);
		reader0->im_w_out(*int_reader_to_writer); 
		reader0->im_h_out(*int2_reader_to_writer);
		reader0->im_name_out(*char_reader_to_writer);
        //name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
        
        conv_layer::conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
         int _stride, int _numFilters, int _pad, ACTIVATION _activation,
         bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords)
        max_layer::max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
        int _stride, bool _crop, int* _inputCoords, int* _outputCoords)
        //void prevLayerCoords(int coords[4], int prevCoords[4], int stride, int filterSize, int prevWidth, int prevHeight, std::string layerType);
        
        int outputCoords[14][4];
        prevLayerCoords(finalOutputCoords, outputCoords[13], 1, 3, 13, 13, "convolutional");
        conv14 = new conv_layer("conv14",14, 13, 13, 512, 1,1,425,1, LINEAR, false, true, outputCoords[13], finalOutputCoords);
        conv14->in(*conv13_to_conv14);
        conv14->out(*conv14_to_region);
        
        prevLayerCoords(outputCoords[13], outputCoords[12], 1, 3, 13, 13, "convolutional");
        conv13 = new conv_layer("conv13",13, 13, 13, 1024, 3, 1, 512, 1, LEAKY, true, true, outputCoords[12], outputCoords[13]);
        conv13->in(*conv12_to_conv13);
        conv13->out(*conv13_to_conv14);
        
        prevLayerCoords(outputCoords[12], outputCoords[11], 1, 3, 13, 13, "convolutional");
        conv12 = new conv_layer("conv12",12, 13, 13, 512, 3,1,1024,1,LEAKY,true, false, NULL, NULL);
        conv12->in(*max11_to_conv12);
        conv12->out(*conv12_to_conv13);
        
        // !!! NOTE !!! this is the only max layer with stride=1
        prevLayerCoords(outputCoords[11], outputCoords[10], 1, 2, 13, 13, "maxpool");
        max11 = new max_layer("max11",11, 13, 13, 512, 2,1, false, NULL, NULL);
        max11->in(*conv10_to_max11);
        max11->out(*max11_to_conv12);
        
        prevLayerCoords(outputCoords[10], outputCoords[9], 1, 3, 26, 26, "convolutional");
        conv10 = new conv_layer("conv10",10, 13, 13, 256, 3,1,512,1, LEAKY, true, false, NULL, NULL);
        conv10->in(*max9_to_conv10);
        conv10->out(*conv10_to_max11);
        
        prevLayerCoords(outputCoords[9], outputCoords[8], 2, 2, 26, 26, "maxpool");
        max9 = new max_layer("max9",9, 26, 26, 256, 2,2, false, NULL, NULL);
        max9->in(*conv8_to_max9);
        max9->out(*max9_to_conv10);

        conv8 = new conv_layer("conv8",8, 26, 26, 128, 3,1,256,1, LEAKY ,true, false, NULL, NULL);
        conv8->in(*max7_to_conv8);
        conv8->out(*conv8_to_max9);
        
        max7 = new max_layer("max7",7, 52, 52, 128, 2,2, false, NULL, NULL);
        max7->in(*conv6_to_max7);
        max7->out(*max7_to_conv8);
        
        conv6 = new conv_layer("conv6",6, 52, 52, 64, 3,1,128,1, LEAKY, true, false, NULL, NULL);
        conv6->in(*max5_to_conv6);
        conv6->out(*conv6_to_max7);

        max5 = new max_layer("max5",5, 104, 104, 64, 2, 2, false, NULL, NULL);
        max5->in(*conv4_to_max5);
        max5->out(*max5_to_conv6);

        conv4 = new conv_layer("conv4",4, 104, 104, 32, 3,1,64,1, LEAKY, true, false, NULL, NULL);
        conv4->in(*max3_to_conv4);
        conv4->out(*conv4_to_max5);
        
        max3 = new max_layer("max3",3, 208, 208, 32, 2,2, false, NULL, NULL);
        max3->in(*conv2_to_max3);
        max3->out(*max3_to_conv4);
        
        conv2 = new conv_layer("conv2",2, 208, 208, 16, 3,1,32, 1, LEAKY, true, false, NULL, NULL);
        conv2->in(*max1_to_conv2);
        conv2->out(*conv2_to_max3);

        max1 = new max_layer("max1",1, 416, 416, 16, 2,2);
        max1->in(*conv0_to_max1);
        max1->out(*max1_to_conv2);
       
        conv0 = new conv_layer("conv0",0, 416, 416, 3, 3,1,16, 1, LEAKY, true, "conv0.weights");
		conv0->in(*in_to_conv0);
		conv0->out(*conv0_to_max1);*/

    }
};
