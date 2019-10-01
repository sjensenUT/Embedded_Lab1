/* 
 *  Embedded System Design & Modeling - Lab 1
 *  (NAMES GO HERE)
 */

#include <vector>
#include <string>
#include <iostream>
#include <systemc.h>
#include <stdio.h>
#include <stdlib.h>
#include "../kahn_process.h"

#include "darknet.h"
#include "array_ops.h"
#include "../../darknet/src/convolutional_layer.h"
#include "../../darknet/src/maxpool_layer.h"
#include "../../darknet/src/region_layer.h"
#include "../../darknet/src/parser.h"
#include "../../darknet/src/activations.h"
#include "../../darknet/src/image.h"

using	std::cout;
using	std::endl;
using std::string;
using std::size_t;
typedef std::vector<std::string> strs;

// These constants are fixed parameters of the YOLO-V2 Tiny network.
const int IMAGE_WIDTH  = 416;
const int IMAGE_HEIGHT = 416;
const int BATCH        = 1;

// Constant float array to hold the anchors for the region layer
// whatever that means
const float ANCHORS[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                           5.47434, 7.88282 , 3.52778, 9.77052, 9.16828};

void	load(int lIdx, const char* attr, float* ptr, int size)
{
	char	fn[100];
	FILE*	fh;

	sprintf(fn, "out/l%i/%s.bin", lIdx, attr);
	fh = fopen(fn, "r");
	fread(ptr, sizeof(float), size, fh);
	fclose(fh);
}

class	image_reader : public kahn_process
{
	public:

	strs	images;

  // Queue data type should be changed to image
	sc_fifo_out<float*> out;
	sc_fifo_out<float*> im_out; 
	sc_fifo_out<int> im_w_out; 
	sc_fifo_out<int> im_h_out; 
	sc_fifo_out<string> im_name_out; 
  	layer l;

	image_reader(sc_module_name name, strs _images)
	:	kahn_process(name),
		images(_images)
	{
		cout << "instantiated image_reader" << endl;
	}

	void	process() override
	{
		for(size_t i=0; i<images.size(); i++)
		{
			cout << "reading image " << images[i] << " @ iter " << iter++ << endl;

			// read images[i] from file
			image orig  = load_image_color( const_cast<char*> (images[i].c_str()), 0, 0);
 			image sized = letterbox_image(orig, IMAGE_WIDTH, IMAGE_HEIGHT);

		      	printf("orig.data[0] : %f\n", (orig.data)[0]);
      			printf("sized.data[0] : %f\n", (sized.data)[0]);

			// sized.data is now the float* that points to the float array that will
			// be the output/input of each layer. The image writer will call free on 
			// this float* to deallocate the data.
			out->write(sized.data);
			im_out->write(orig.data);
			im_w_out->write(orig.w);
			im_h_out->write(orig.h); //give both width in height in queue of length 2
			char name[10];
			sprintf(name,"image%zu",i);
      string name_str(name);			
      im_name_out->write(name_str);
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
	const	ACTIVATION activation;
	const	bool batchNormalize;
	
  	sc_fifo_in<float*> in;
	sc_fifo_out<float*> out;

  	convolutional_layer l;

	conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
             int _stride, int _numFilters, int _pad, ACTIVATION _activation,
             bool _batchNormalize, const char* _weightsFileName)
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
		int groups  = 1;
    		// Padding is 0 by default. If PAD is true (non-zero), then it equals half the 
    		// filter size rounding down (see parse_convolutional() in darknet's parser.c)
    		int padding = 0;
    		if (this->pad != 0) {
      			padding = this->filterSize / 2;
    		}

    		// Call make_convolutional_layer() to create the layer object
    		l = make_convolutional_layer(BATCH, _h, _w, _c, this->numFilters, groups,
          		this->filterSize, this->stride, padding, activation, (int) batchNormalize,
          		0, 0, 0);  
 
 		// Load the weights into the layer
    		//FILE* weightsFile = fopen(_weightsFileName, "r");
    		//if(weightsFile) {
      		//	load_convolutional_weights(l, weightsFile);   
    		//} else {
      		//	cout << "Could not find weights file " << _weightsFileName << endl;
    		//}

		//new code for loading weights, copied from kamyar
		int num = l.c/l.groups*l.n*l.size*l.size;
		cout << "l.size = " << l.size << endl;
		load(layerIndex, "biases", l.biases, l.n);

		if(l.batch_normalize)
		{
			load(layerIndex, "scales", l.scales, l.n);
			load(layerIndex, "mean",   l.rolling_mean, l.n);
			load(layerIndex, "variance", l.rolling_variance, l.n);
		}

		load(layerIndex, "weights", l.weights, num);

		printf("loaded parameters of layer %i\n", layerIndex);
    		printf("Biases : %f %f %f ...\n", l.biases[0], l.biases[1], l.biases[2]);
    		printf("Weights: %f %f %f ...\n", l.weights[0], l.weights[1], l.weights[2]);
  	}

	void	process() override
	{
		float* input;

    		// Read the output from the previos layer
		in->read(input);
 
		cout << "forwarding convolutional layer " << layerIndex << " @ iter " << iter << endl;

    		// Create a dummy network object. forward_convolutional_layer only uses the "input"
    		// and "workspace" elements of the network struct. "input" is simply the output of
    		// the previous layer, while "workspace" points to an array of floats that we will
    		// create just before calling. The size can be determined by layer.get_workspace_size().
    		network dummyNetwork;
		dummyNetwork.input = input;

		printf("inputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", input[j]);
    }
    printf("\n");
	
		cout << "getting workspace size" << endl; 
    		size_t workspace_size = get_convolutional_workspace_size(l);
		cout << "allocating workspace memory" << endl; 
    		dummyNetwork.workspace = (float*) calloc(1, workspace_size);
		cout << "forward convoluting" << endl;
    		forward_convolutional_layer(l, dummyNetwork);
	
	  printf("outputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", l.output[j]);
    }
    printf("\n");
	
   		cout << "freeing" << endl;
		free(dummyNetwork.workspace);
    		// Send off the layer's output to the next layer!
		out->write(l.output);

    		// Now we're done with the workspace - deallocate it or else memory leaks.
		//free(dummyNetwork.input);
		//free(input); 

	}
};

class	max_layer : public kahn_process
{
	public:

	const	int stride;
	const	int layerIndex;
	const	int filterSize;	
	
	sc_fifo_in<float*> in;
	sc_fifo_out<float*> out;

  	layer l; 
	max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
            int _stride) 
	:	kahn_process(name),
		stride(_stride),
		layerIndex(_layerIndex),
		filterSize(_filterSize)
	{
		cout << "instantiated max layer " << layerIndex << " with filter size of " << filterSize << " and stride of " << stride << endl;

    	// Create the underlying darknet layer
    	l = make_maxpool_layer(BATCH, _h, _w, _c, this->filterSize, 
                           this->stride, filterSize-1);
    
	}

	void	process() override
	{
		float* data;

		in->read(data);
		cout << "forwarding max layer " << layerIndex << " @ iter " << iter << endl;

		printf("inputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", data[j]);
    }
    printf("\n");

   	// Call forward_maxpool_layer() here, read from layer.output and write to out
   	// Create a dummy network object. The function only uses network.input
   	network dummyNetwork;
  	dummyNetwork.input = data;
   	forward_maxpool_layer(l, dummyNetwork);

	  printf("outputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", l.output[j]);
    }
    printf("\n");
  
  	out->write(l.output);	
	}
};

// Necessary? Not sure yolov2-tiny has a "detection" layer, whatever that is.
class	region_layer : public kahn_process
{
	public:
	
	const float* anchors;
	const bool biasMatch;
	const int classes;
	const int coords;
	const int num;
	const bool softMax;
	const float jitter;
	const bool rescore;
	const int objScale;
	const bool noObjectScale;
	const int classScale;
	const int coordScale;
	const bool absolute;
	const float thresh;
	const bool random;
	
	sc_fifo_in<float*> in;
//	sc_fifo_out<float*> out;
//	sc_fifo_out<int>  l_out; // this is to pass the l.classes parameter for draw_detections
	
	sc_fifo_in<float*> im_in; 
	sc_fifo_in<int> im_w_in; // for width and height of image
	sc_fifo_in<int> im_h_in; 
	sc_fifo_in<string> im_name_in; 

	image ** alphabets; 
	layer l;	

	region_layer(sc_module_name name, float _anchors[], bool _biasMatch, int _classes,
               int _coords, int _num, bool _softMax, float _jitter, bool _rescore, 
               int _objScale, bool _noObjectScale, int _classScale, int _coordScale,
               bool _absolute, float _thresh, bool _random, int _w, int _h) 
	:	kahn_process(name),
		anchors(_anchors),
		biasMatch(_biasMatch),
		classes(_classes),
		coords(_coords),
		num(_num),
		softMax(_softMax),
		jitter(_jitter),
		rescore(_rescore),
		objScale(_objScale),
		noObjectScale(_noObjectScale),
		classScale(_classScale),
		coordScale(_coordScale),
		absolute(_absolute),
		thresh(_thresh),
		random(_random)
	{
		cout << "instantiating region layer" << endl;
		l = make_region_layer(BATCH, _w, _h, this->num, this->classes, this->coords);
    l.log        = 0;
    l.sqrt       = 0;
    l.softmax    = softMax;
    l.background = 0;
    l.max_boxes  = 30;
    l.jitter     = jitter;
    l.rescore    = rescore;
    l.thresh     = thresh;
    l.classfix   = 0;
    l.absolute   = absolute;
    l.random     = random;
    l.coord_scale = coordScale;
    l.object_scale = objScale;
    l.noobject_scale = noObjectScale;
    l.mask_scale = 1;
    l.class_scale = classScale;;
    l.bias_match = biasMatch;
		alphabets = load_alphabet(); 
	}

	void	process() override
	{
		float* data;
		string image_name; 
		image im; 
	
		in->read(data);
		im_name_in->read(image_name);
		im_in->read(im.data);
		im_w_in->read(im.w);
		im_h_in->read(im.h); 
	  im.c = 3;

		cout << "forwarding detection layer @ iter " << iter << endl;
 
  	network dummyNetwork;
	 	dummyNetwork.input = data;
		forward_region_layer(l, dummyNetwork);

	  printf("outputs of region layer, are");
    for(int j = 0; j < 10; j++){
        printf(" %f", l.output[j]);
    }
    printf("\n");

		// NOTE: this threshold value is NOT the same thing as l.thresh
		// This comes from the -thresh flag specified when running darknet's
		// detector example. The layer's threshold (l.thresh) comes from the 
		// cfg file.
    float det_thresh = 0.5;
		int nboxes = 0; 
			
 		cout << "attempting to detect" << endl;
		
    list *options = read_data_cfg("../../darknet/cfg/yolov2-tiny.cfg");
    char *name_list = option_find_str(options, "names", "../../darknet/data/coco.names");
    char **names = get_labels(name_list);
		int i;
		int w = im.w;
		int h = im.h; 	
    //get_network_boxes -> make network boxes
		int * map = nullptr; 
		int relative = 1; 

		// get network boxes -> make network boxes -> num detections
				

		if(l.type == DETECTION || l.type == REGION){
			nboxes += l.w*l.h*l.n;
		}

   		detection *dets = (detection *) calloc(nboxes, sizeof(detection));
   		for(i = 0; i < nboxes; ++i){
     			dets[i].prob = (float *) calloc(l.classes, sizeof(float));
      			if(l.coords > 4){
		    		dets[i].mask = (float *) calloc(l.coords-4, sizeof(float));
      			}
  		}
						
		cout << "get region detections " << endl;
		get_region_detections(l, w, h, IMAGE_WIDTH, IMAGE_HEIGHT, det_thresh, map, 0.5, relative, dets);
    //dets += l.w*l.h*l.n;
		
  	// draw detections
	  cout << "draw detections" << endl;
    float nms = 0.45;
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms); 
		draw_detections(im, dets, nboxes, det_thresh, names, alphabets, l.classes);

	 	cout << "free detections" << endl; 
		free_detections(dets, nboxes); 
				
		cout << "attempting to write" << endl; 
			
		// dump to file
		char outFN[100];
		sprintf(outFN,"%s_testOut",image_name.c_str()); 
		cout << "Saving image" << endl;
    save_image(im,outFN);
		free_image(im); 
		cout << "writing predictions to " << outFN << "  @ iter " << iter++ << endl;
		//free(alphabets);  Now part of the constructor and I don't free it here? 
	}

};
	
	

// Might need to make separate class for "region" layer

class	kpn_neuralnet : public sc_module
{
	public:
	
  // Declare all queues between our layers here
  // I think the data type for all of them will be image
	sc_fifo<float*>	*reader_to_conv0, 
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
			*conv14_to_region;
//			*region_to_writer;

	sc_fifo<float*>  *reader_to_writer; 
//	sc_fifo<int>    *layer_region_to_writer; 
	sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	sc_fifo<string>  *char_reader_to_writer; 
			//*conv2_to_detection,
			//*detection_to_writer;

  // Declare all layers here
	max_layer	*max1, *max3, *max5, *max7, *max9, *max11;
	conv_layer	*conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;
	region_layer	*region;
	image_reader	*reader0;
//	image_writer	*writer0;

  // Constructor of the overall network. Initialize all queues and layers
	kpn_neuralnet(sc_module_name name) : sc_module(name)
	{
		//strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
		strs images = {"../../darknet/data/dog.jpg"};
		//std::string cfgFile = "../../darknet/cfg/yolov2-tiny.cfg";
		//std::string weightFile = "../../darknet/yolov2-tiny.weights";
		//char *cfgFileC = new char[cfgFile.length() + 1];
		//strcpy(cfgFileC, cfgFile.c_str());
		//char *weightFileC = new char[weightFile.length() + 1];
		//strcpy(weightFileC, weightFile.c_str());
		//network *net = load_network(cfgFileC, weightFileC, 0);
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
//		region_to_writer 	= new sc_fifo<float*>(1);

		reader_to_writer 	= new sc_fifo<float*>(1); 
		int_reader_to_writer	= new sc_fifo<int>(1); // needed to send im.w and im.h
		int2_reader_to_writer 	= new sc_fifo<int>(1); 
		char_reader_to_writer  	= new sc_fifo<string>(1);
//		layer_region_to_writer 	= new sc_fifo<int>(1)
//		reader_to_writer 	= new sc_fifo<float*>(1); 
//		int_reader_to_writer	= new sc_fifo<int>(2); // needed to send im.w and im.h
//		layer_region_to_writer 	= new sc_fifo<int>(1);
		
    // Here is where we will indicate the parameters for each layer. These can
    // be found in the cfg file for yolov2-tiny in the darknet folder.
		reader0 = new image_reader("image_reader",images);
		reader0->out(*reader_to_conv0);
		reader0->im_out(*reader_to_writer);
		reader0->im_w_out(*int_reader_to_writer); 
		reader0->im_h_out(*int2_reader_to_writer);
		reader0->im_name_out(*char_reader_to_writer);
		//name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
		conv0 = new conv_layer("conv0",0, 416, 416, 3, 3,1,16, 1, LEAKY, true, "conv0.weights");
		conv0->in(*reader_to_conv0);
		conv0->out(*conv0_to_max1);

		max1 = new max_layer("max1",1, 416, 416, 16, 2,2);
		max1->in(*conv0_to_max1);
		max1->out(*max1_to_conv2);

		conv2 = new conv_layer("conv2",2, 208, 208, 16, 3,1,32, 1, LEAKY, true, "conv2.weights");
		conv2->in(*max1_to_conv2);
		conv2->out(*conv2_to_max3);

		
		max3 = new max_layer("max3",3, 208, 208, 32, 2,2);
                max3->in(*conv2_to_max3);
                max3->out(*max3_to_conv4);
		
		conv4 = new conv_layer("conv4",4, 104, 104, 32, 3,1,64,1, LEAKY, true, "conv4.weights");
                conv4->in(*max3_to_conv4);
                conv4->out(*conv4_to_max5);
		
		max5 = new max_layer("max5",5, 104, 104, 64, 2, 2);
                max5->in(*conv4_to_max5);
                max5->out(*max5_to_conv6);
		
		conv6 = new conv_layer("conv6",6, 52, 52, 64, 3,1,128,1, LEAKY, true, "conv6.weights");
                conv6->in(*max5_to_conv6);
                conv6->out(*conv6_to_max7);
	
		max7 = new max_layer("max7",7, 52, 52, 128, 2,2);
                max7->in(*conv6_to_max7);
                max7->out(*max7_to_conv8);
		
		conv8 = new conv_layer("conv8",8, 26, 26, 128, 3,1,256,1, LEAKY ,true, "conv8.weights");
                conv8->in(*max7_to_conv8);
                conv8->out(*conv8_to_max9);
		
		max9 = new max_layer("max9",9, 26, 26, 256, 2,2);
                max9->in(*conv8_to_max9);
                max9->out(*max9_to_conv10);
		
		conv10 = new conv_layer("conv10",10, 13, 13, 256, 3,1,512,1, LEAKY, true, "conv10.weights");
                conv10->in(*max9_to_conv10);
                conv10->out(*conv10_to_max11);
		
		max11 = new max_layer("max11",11, 13, 13, 512, 2,1);
                max11->in(*conv10_to_max11);
                max11->out(*max11_to_conv12);

		conv12 = new conv_layer("conv12",12, 13, 13, 512, 3,1,1024,1,LEAKY,true, "conv12.weights");
                conv12->in(*max11_to_conv12);
                conv12->out(*conv12_to_conv13);
		
		conv13 = new conv_layer("conv13",13, 13, 13, 1024, 3,1,512,1,LEAKY,true,"conv13.weights");
                conv13->in(*conv12_to_conv13);
                conv13->out(*conv13_to_conv14);

		conv14 = new conv_layer("conv14",14, 13, 13, 512, 1,1,425,1, LINEAR, false, "conv14.weights");
                conv14->in(*conv13_to_conv14);
                conv14->out(*conv14_to_region);

		region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                               true, 1, 1, true, 0.6, true, 13, 13);
		region->in(*conv14_to_region);
		region->im_in(*reader_to_writer);
		region->im_w_in(*int_reader_to_writer); 
		region->im_h_in(*int2_reader_to_writer);
		region->im_name_in(*char_reader_to_writer); 
//		region->out(*region_to_writer);
//		region->l_out(*layer_region_to_writer);

		/*		
		writer0 = new image_writer("image_writer",images);
		writer0->in(*region_to_writer);
		writer0->l_in(*layer_region_to_writer); 
		writer0->im_in(*reader_to_writer);
		writer0->imd_in(*int_reader_to_writer); 
		 */
	}
};



// This will probably remain as-is.
int	sc_main(int argc, char * argv[]) 
{
	kpn_neuralnet knn0("kpn_neuralnet");
	sc_start();
	return 0;
}
