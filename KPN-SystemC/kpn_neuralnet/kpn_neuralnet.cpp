/* 
 *  Embedded System Design & Modeling - Lab 1
 *  (NAMES GO HERE)
 */

#include <vector>
#include <string>
#include <iostream>
#include <systemc.h>
#include "../kahn_process.h"
#include "../../darknet/include/darknet.h"
using	std::cout;
using	std::endl;
typedef std::vector<std::string> strs;

class	image_reader : public kahn_process
{
	public:

	strs	images;

  // Queue data type should be changed to image
	sc_fifo_out<float> out;
  layer l;

	image_reader(sc_module_name name, strs _images)
	:	kahn_process(name),
		images(_images)
	{
		cout << "instantiated image_reader" << endl;
	}

	void	process() override
	{
		float	val = 1.234;

		for(size_t i=0; i<images.size(); i++)
		{
			cout << "reading image " << images[i] << " @ iter " << iter++ << endl;

			// read images[i] from file
			// Call load_image_color() and letterbox_image() here for each image,
			// then write it to output queue
			// for(val in images[i])
			// 	out->write(val);
			out->write(val);
		}
	}
}; 

class	image_writer : public kahn_process
{
	public:

	int	iter;

	strs	images;

  // Queue data type should be changed to image
	sc_fifo_in<float> in;

	image_writer(sc_module_name name, strs _images)
	:	kahn_process(name),
		images(_images)
	{
		cout << "instantiated image_writer" << endl;
	}

	void	process() override
	{
		float	    val;
		std::string outFN;

		for(size_t i=0; i<images.size(); i++)
		{
			// read values from "in"
			in->read(val);

			// dump to file
			outFN = "predicted_";
			outFN += images[i];

			cout << "writing predictions to " << outFN << "  @ iter " << iter++ << endl;
		}
	}
};

class	conv_layer : public kahn_process
{
	public:

	const	int stride;
	const	int numFilters;
	const	int layerIndex;
	const	int filterSize;	
	const	int pad;
	const	std::string activation;
	const	bool batchNormalize;
  // Queue data type should be changed to image
	sc_fifo_in<float> in;
	sc_fifo_out<float> out;

  // Store a layer object: layer l;

	conv_layer(sc_module_name name, int _layerIndex, int _filterSize, int _stride, int _numFilters, int _pad, std::string _activation, bool _batchNormalize)
	:	kahn_process(name),
		stride(_stride),
		numFilters(_numFilters),
		layerIndex(_layerIndex),
		filterSize(_filterSize),
		pad(_pad),
		activation(_activation),
		batchNormalize(_batchNormalize)
	{
		cout << "instantiated convolutional layer " << layerIndex << " with filter size of " << filterSize << ", stride of " << stride << " and " << numFilters << " filters" << endl;

    // Call make_convolutional_layer() to create the layer object, store it inside this
    // object. Figure out what values to pass to make_convolutional_layer() that are 
    // not parameters to this constructor.

	}

	void	process() override
	{
		float val;

		in->read(val);
		cout << "forwarding convolutional layer " << layerIndex << " @ iter " << iter << endl;

    // Modified forward_convolutional_layer() code goes here
    // Or we define it as a private method and call it from here.
    // Then write layer.output to out queue

		out->write(10*val);
	}
};

class	max_layer : public kahn_process
{
	public:

	const	int stride;
	const	int layerIndex;
	const	int filterSize;	

	sc_fifo_in<float> in;
	sc_fifo_out<float> out;

  // Layer object goes here

	max_layer(sc_module_name name, int _layerIndex, int _filterSize, int _stride) 
	:	kahn_process(name),
		stride(_stride),
		layerIndex(_layerIndex),
		filterSize(_filterSize)
	{
		cout << "instantiated max layer " << layerIndex << " with filter size of " << filterSize << " and stride of " << stride << endl;

    // Call make_maxpool_layer() here
    
	}

	void	process() override
	{
		float val;

		in->read(val);
		cout << "forwarding max layer " << layerIndex << " @ iter " << iter << endl;
    
    // Call forward_maxpool_layer() here, read from layer.output and write to out
  
		out->write(val+1.5);
	}
};

// Necessary? Not sure yolov2-tiny has a "detection" layer, whatever that is.
class	detection_layer : public kahn_process
{
	public:

	sc_fifo_in<float> in;
	sc_fifo_out<float> out;

	detection_layer(sc_module_name name) 
	:	kahn_process(name)
	{
		cout << "instantiated detection layer " << endl;
	}

	void	process() override
	{
		float val;

		in->read(val);
		cout << "forwarding detection layer @ iter " << iter << endl;
		out->write(val+0.1);
	}
};

// Might need to make separate class for "region" layer

class	kpn_neuralnet : public sc_module
{
	public:

  // Declare all queues between our layers here
  // I think the data type for all of them will be image
	sc_fifo<float>	*reader_to_conv0, 
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
			*conv14_to_region,
			*region_to_writer;
			//*conv2_to_detection,
			//*detection_to_writer;

  // Declare all layers here
	max_layer	*max1, *max3, *max5, *max7, *max9, *max11;
	conv_layer	*conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;
	image_reader	*reader0;
	image_writer	*writer0;
	//detection_layer	*det0;

  // Constructor of the overall network. Initialize all queues and layers
	kpn_neuralnet(sc_module_name name) : sc_module(name)
	{
		strs images = {"dog.jpg", "horse.jpg"};
		//std::string cfgFile = "../../darknet/cfg/yolov2-tiny.cfg";
		//std::string weightFile = "../../darknet/yolov2-tiny.weights";
		//char *cfgFileC = new char[cfgFile.length() + 1];
		//strcpy(cfgFileC, cfgFile.c_str());
		//char *weightFileC = new char[weightFile.length() + 1];
		//strcpy(weightFileC, weightFile.c_str());
		//network *net = load_network(cfgFileC, weightFileC, 0);
		reader_to_conv0 = new sc_fifo<float>(1);
		conv0_to_max1   = new sc_fifo<float>(1);
		max1_to_conv2   = new sc_fifo<float>(1);
		conv2_to_max3   = new sc_fifo<float>(1);
                max3_to_conv4   = new sc_fifo<float>(1);
		conv4_to_max5   = new sc_fifo<float>(1);
                max5_to_conv6   = new sc_fifo<float>(1);
		conv6_to_max7   = new sc_fifo<float>(1);
                max7_to_conv8   = new sc_fifo<float>(1);
		conv8_to_max9   = new sc_fifo<float>(1);
                max9_to_conv10   = new sc_fifo<float>(1);
		conv10_to_max11   = new sc_fifo<float>(1);
                max11_to_conv12   = new sc_fifo<float>(1);
		conv12_to_conv13   = new sc_fifo<float>(1);
                conv13_to_conv14   = new sc_fifo<float>(1);
		conv14_to_region   = new sc_fifo<float>(1);
		region_to_writer = new sc_fifo<float>(1);

    // Here is where we will indicate the parameters for each layer. These can
    // be found in the cfg file for yolov2-tiny in the darknet folder.
		reader0 = new image_reader("image_reader",images);
		reader0->out(*reader_to_conv0);
		//name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
		conv0 = new conv_layer("conv0",0,3,1,16, 1, "leaky", true);
		conv0->in(*reader_to_conv0);
		conv0->out(*conv0_to_max1);

		max1 = new max_layer("max1",1,2,2);
		max1->in(*conv0_to_max1);
		max1->out(*max1_to_conv2);

		conv2 = new conv_layer("conv2",2,3,1,32, 1,"leaky", true);
		conv2->in(*max1_to_conv2);
		conv2->out(*conv2_to_max3);
		
		max3 = new max_layer("max3",3,2,2);
                max3->in(*conv2_to_max3);
                max3->out(*max3_to_conv4);
		
		conv4 = new conv_layer("conv4",4,3,1,64,1,"leaky",true);
                conv4->in(*max3_to_conv4);
                conv4->out(*conv4_to_max5);
		
		max5 = new max_layer("max5",5,2,2);
                max5->in(*conv4_to_max5);
                max5->out(*max5_to_conv6);
		
		conv6 = new conv_layer("conv6",6,3,1,128,1,"leaky",true);
                conv6->in(*max5_to_conv6);
                conv6->out(*conv6_to_max7);
	
		max7 = new max_layer("max7",7,2,2);
                max7->in(*conv6_to_max7);
                max7->out(*max7_to_conv8);
		
		conv8 = new conv_layer("conv8",8,3,1,256,1,"leaky",true);
                conv8->in(*max7_to_conv8);
                conv8->out(*conv8_to_max9);
		
		max9 = new max_layer("max9",9,2,2);
                max9->in(*conv8_to_max9);
                max9->out(*max9_to_conv10);
		
		conv10 = new conv_layer("conv10",10,3,1,512,1,"leaky",true);
                conv10->in(*max9_to_conv10);
                conv10->out(*conv10_to_max11);
		
		max11 = new max_layer("max11",11,2,2);
                max11->in(*conv10_to_max11);
                max11->out(*max11_to_conv12);

		conv12 = new conv_layer("conv12",12,3,1,1024,1,"leaky",true);
                conv12->in(*max11_to_conv12);
                conv12->out(*conv12_to_conv13);
		
		conv13 = new conv_layer("conv13",13,3,1,512,1,"leaky",true);
                conv13->in(*conv12_to_conv13);
                conv13->out(*conv13_to_conv14);

		conv14 = new conv_layer("conv14",14,1,1,425,1,"linear",false);
                conv14->in(*conv13_to_conv14);
                conv14->out(*conv14_to_region);
		//TODO: Implement region layer
		//det0 = new detection_layer("detection");
		//det0->in(*conv2_to_detection);
		//det0->out(*detection_to_writer);

		writer0 = new image_writer("image_writer",images);
		writer0->in(*region_to_writer);
	}
};

// This will probably remain as-is.
int	sc_main(int, char *[]) 
{
	kpn_neuralnet knn0("kpn_neuralnet");
	sc_start();
	return 0;
}
