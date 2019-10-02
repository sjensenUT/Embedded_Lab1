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
#include "merge_scatter.h"
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


void getTileCoords(int width, int height, int coords[9][4]){
    for(int i = 0; i < 3; i++){ // TILE ROW
        for(int j = 0; j < 3; j++){ // TILE COL
            //coords[i*3 + j] = new int[4] {j*width/3, i*height/3, (j+1)*width/3 - 1, (i+1)*height/3 - 1};
            coords[i*3 + j][0] = j*width/3;
            coords[i*3 + j][1] = i*height/3;
            coords[i*3 + j][2] = (j == 2) ? width-1  : (j+1)*width/3 - 1;
            coords[i*3 + j][3] = (i == 2) ? height-1 : (i+1)*height/3 - 1;
        }
    }
}

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


float get_pixel(image m, int x, int y, int c)
{
  return m.data[c*m.h*m.w + y*m.w + x];
}

void printChannels(image m, int x, int y)
{
  printf("(%d,%d): ", x, y);
  for (int c = 0; c < m.c; c++) {
    cout << get_pixel(m, x, y, c) << " ";
  }
  cout << endl;
}

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
    const bool crop;
	int* inputCoords;
    int* outputCoords;


    sc_fifo_in<float*> in;
	sc_fifo_out<float*> out;

    convolutional_layer l;


    void printCoords() {
        printf("Layer %d input coords: %d %d %d %d\n", layerIndex,
            this->inputCoords[0],
            this->inputCoords[1],
            this->inputCoords[2],
            this->inputCoords[3]);  
    }


	conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
             int _stride, int _numFilters, int _pad, ACTIVATION _activation,
             bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords)
	:	kahn_process(name),
		stride(_stride),
		numFilters(_numFilters),
		layerIndex(_layerIndex),
		filterSize(_filterSize),
		pad(_pad),
		activation(_activation),
		batchNormalize(_batchNormalize),
        crop(_crop)

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
 
		//new code for loading weights, copied from kamyar
		int num = l.c/l.groups*l.n*l.size*l.size;
		//cout << "l.size = " << l.size << endl;
		load(layerIndex, "biases", l.biases, l.n);

		if(l.batch_normalize)
		{
			load(layerIndex, "scales", l.scales, l.n);
			load(layerIndex, "mean",   l.rolling_mean, l.n);
			load(layerIndex, "variance", l.rolling_variance, l.n);
		}

		load(layerIndex, "weights", l.weights, num);

//		printf("loaded parameters of layer %i\n", layerIndex);
//   	printf("Biases : %f %f %f ...\n", l.biases[0], l.biases[1], l.biases[2]);
//    printf("Weights: %f %f %f ...\n", l.weights[0], l.weights[1], l.weights[2]);

    // Copy the input and output coordinates.
        if (crop) {
            inputCoords = new int[4];
            outputCoords = new int[4];
            for (int j = 0; j < 4; j++) {
                inputCoords[j] = _inputCoords[j];
                outputCoords[j] = _outputCoords[j];
            }
            printCoords();
        }

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

//  	printf("inputs of layer %d, are", layerIndex);
//    for(int j = 0; j < 10; j++){
//        printf(" %f", input[j]);
//    }
//    printf("\n");
	
        size_t workspace_size = get_convolutional_workspace_size(l);
        dummyNetwork.workspace = (float*) calloc(1, workspace_size);
        forward_convolutional_layer(l, dummyNetwork);
	
//	  printf("outputs of layer %d, are", layerIndex);
//    for(int j = 0; j < 10; j++){
//        printf(" %f", l.output[j]);
//    }
//    printf("\n");
	
		free(dummyNetwork.workspace);

    float* outputImage = l.output;

    // Now it's time to crop the data if this layer is configured to do cropping.
    if (crop) {
        
        // Calculate the relative coordinates for cropping
        int cropped_width = outputCoords[2] - outputCoords[0] + 1;
        int cropped_height = outputCoords[3] - outputCoords[1] + 1;
        int left_crop   = outputCoords[0] - inputCoords[0];
        int top_crop    = outputCoords[1] - inputCoords[1];
        int right_crop  = inputCoords[2]  - outputCoords[2];
        int bottom_crop = inputCoords[3]  - outputCoords[3];
        int cropCoords[4] = { left_crop, top_crop,
                              left_crop + cropped_width - 1,
                              top_crop + cropped_height - 1};

//        printf("Cropping tile %d in layer %d\n", tileNumber, layerIndex);
//        printf("Cropping image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
//               inputCoords[0], inputCoords[1], inputCoords[2], inputCoords[3],
//               outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
//        printf("Crop amounts (left, top, right, bottom): (%d, %d, %d, %d)\n", 
//               left_crop, top_crop, right_crop, bottom_crop);
//        printf("Relative crop coordinates are (%d, %d) (%d, %d)\n",
//                cropCoords[0], cropCoords[1], cropCoords[2], cropCoords[3]);
        outputImage = getSubArray(l.output, cropCoords, l.w, l.h, numFilters);

        //printf("Relative crop coordinates 2 are (%d, %d) (%d, %d)\n",
        //    cropCoords2[0], cropCoords2[1], cropCoords2[2], cropCoords2[3]);
                         
        printf("Cropping tile %d\n", layerIndex);
        printf("l.c = %d\n", l.c);
        printf("l.w = %d\n", l.w);
        printf("l.h = %d\n", l.h);
        printf("this->numFilters = %d\n", this->numFilters);

        printf("Cropping image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
                inputCoords[0], inputCoords[1], inputCoords[2], inputCoords[3],
                outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
        outputImage = getSubArray(l.output, cropCoords, l.w, l.h, this->numFilters);
    }
    // Send off the layer's output to the next layer!
		out->write(outputImage);
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

//        printf("inputs of layer %d, are", layerIndex);
//        for(int j = 0; j < 10; j++){
//            printf(" %f", data[j]);
//        }
//        printf("\n");

   	    // Call forward_maxpool_layer() here, read from layer.output and write to out
   	    // Create a dummy network object. The function only uses network.input
   	    network dummyNetwork;
  	    dummyNetwork.input = data;
   	    forward_maxpool_layer(l, dummyNetwork);

	    /*printf("outputs of layer %d, are", layerIndex);
        for(int j = 0; j < 10; j++){
            printf(" %f", l.output[j]);
        }
        printf("\n");*/
  
  	    out->write(l.output);	
	}
};

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

    for (int i = 0; i < 10; i++) {
        l.biases[i] = ANCHORS[i];   
    }
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

//	  printf("outputs of region layer, are");
//    for(int j = 0; j < 10; j++){
//        printf(" %f", l.output[j]);
//    }
//    printf("\n");

		// NOTE: this threshold value is NOT the same thing as l.thresh
		// This comes from the -thresh flag specified when running darknet's
		// detector example. The layer's threshold (l.thresh) comes from the 
		// cfg file.
    float det_thresh = 0.5;
		int nboxes = 0; 
		
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
						
		get_region_detections(l, w, h, IMAGE_WIDTH, IMAGE_HEIGHT, det_thresh, map, 0.5, relative, dets);
		
  	// draw detections
    float nms = 0.45;
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms); 
		draw_detections(im, dets, nboxes, det_thresh, names, alphabets, l.classes);

		free_detections(dets, nboxes); 
				
		// dump to file
		char outFN[100];
		sprintf(outFN,"%s_testOut",image_name.c_str()); 
        save_image(im,outFN);
		free_image(im); 
		cout << "writing predictions to " << outFN << "  @ iter " << iter++ << endl;
		//free(alphabets);  Now part of the constructor and I don't free it here? 
	}
};

int coerce (int val, int min, int max) {
  if (val < min) return min;
  if (val > max) return max;
  return val;
}

class   conv_layer_unfused : public sc_module
{
    public:
    sc_fifo<float*> *scatter_to_conv[9],
        *conv_to_merge[9];

    scatter_layer *scatter;
    conv_layer *conv[9];
    merge_layer *merge;
    conv_layer_unfused(sc_module_name name, int layerIndex, int coords[][4],
                       int inputWidth, int inputHeight, int c, int filterSize, int stride, int numFilters, int pad,
                       ACTIVATION activation, bool batchNormalize) : sc_module(name)
    {
        cout << "instantiating fused conv layer" << endl;
        cout << "inputWidth = " << inputWidth << ", inputHeight = " << inputHeight << ", c = " << c << endl; 
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 4; j++){
                cout << "coords[" << i << "][" << j << "] = " << coords[i][j]  << endl;
            }
        } 
        int *widths = new int[3]  { coords[0][2] - coords[0][0] + 1,
                                    coords[1][2] - coords[1][0] + 1,
                                    coords[2][2] - coords[2][0] + 1 };
        int *heights = new int[3] { coords[0][3] - coords[0][1] + 1,
                                    coords[3][3] - coords[3][1] + 1,
                                    coords[6][3] - coords[6][1] + 1 };
        //int outputWidth = widths[0] + widths[1] + widths[2];
        //int outputHeight = heights[0] + heights[1] + heights[2];

        // Determine how much padding we should use for each tile.
        //int padding = 0;
        //if (pad) padding = filterSize / 2;
        
        // Create the padded coordinates
        if(pad){
            int paddedCoords[9][4];
            calcPrevCoords(coords, paddedCoords, stride, filterSize, inputWidth, inputHeight, "convolutional");
            for (int j = 0; j < 9; j++) {
                /*// Top-left X coordinate
                paddedCoords[j][0] = coerce(coords[j][0] - padding, 0, totalWidth); 
                // Top-left Y coordinate
                paddedCoords[j][1] = coerce(coords[j][1] - padding, 0, totalHeight); 
                // Bottom-right X coordinate
                paddedCoords[j][2] = coerce(coords[j][2] + padding, 0, totalWidth); 
                // Bottom-right Y coordinate
                paddedCoords[j][3] = coerce(coords[j][3] + padding, 0, totalHeight);*/ 
                printf("Tile %d padded coordinates: (%d, %d) (%d, %d)\n", j,
                    paddedCoords[j][0], paddedCoords[j][1],
                    paddedCoords[j][2], paddedCoords[j][3]);
                //cout << "in conv_layer instantiation loop i = " << i << endl;
                int w = paddedCoords[j][2] - paddedCoords[j][0] + 1;
                int h = paddedCoords[j][3] - paddedCoords[j][1] + 1;
                conv[j] = new conv_layer("conv", layerIndex, w, h, c, filterSize, stride, numFilters, pad, activation, batchNormalize,
                    true, paddedCoords[j], coords[j]);
            }
            scatter = new scatter_layer("scatter", paddedCoords, inputWidth, inputHeight, c);

        }else{
            for(int j = 0; j < 9; j++){
                //cout << "in conv_layer instantiation loop i = " << i << endl;
                int w = coords[j][2] - coords[j][0] + 1;
                int h = coords[j][3] - coords[j][1] + 1;
                conv[j] = new conv_layer("conv", layerIndex, w, h, c, filterSize, stride, numFilters, pad, activation, batchNormalize,
                                      false, NULL, NULL);
            }
            scatter = new scatter_layer("scatter", coords, inputWidth, inputHeight, c);

        }
        //cout << "beginning merge instantiation" << endl;
        merge = new merge_layer("merge", widths, heights, numFilters);
        //cout << "finished instantiating merge layer" << endl;
        for(int i = 0; i < 9; i++){
            //cout << "in fifo assignment loop i = " << i << endl;
            //cout << "assigning scatter_to_conv" << endl;
            scatter_to_conv[i] = new sc_fifo<float*>(1);
            //cout << "assigning conv_to_merge" << endl;
            conv_to_merge[i] = new sc_fifo<float*>(1);
            //cout << "assigning conv.in" << endl;
            conv[i]->in(*scatter_to_conv[i]);
            //cout << "assigning conv.out" << endl;
            conv[i]->out(*conv_to_merge[i]);
            //cout << "assigning scatter.out" << endl;
            scatter->out[i](*scatter_to_conv[i]);
            //cout << "assigning merge.in" << endl;
            merge->in[i](*conv_to_merge[i]);
        }
    }
};

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
	conv_layer	*conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;
	conv_layer_unfused *conv0;
    region_layer	*region;
	image_reader	*reader0;
//	image_writer	*writer0;

  // Constructor of the overall network. Initialize all queues and layers
	kpn_neuralnet(sc_module_name name) : sc_module(name)
	{
		strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
		//strs images = {"../../darknet/data/dog.jpg"};
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
        int tileCoords[9][4];
        getTileCoords(416, 416, tileCoords);
        //int testCoords[][] = new int[][2];
        //testCoords[0] = new int[2] {1, 2};
        //testCoords[1] = new int[2] {3, 4};
        conv0 = new conv_layer_unfused("conv0", 0, tileCoords, 416, 416, 3, 3, 1, 16, 1,  LEAKY, true);
        conv0->scatter->in(*reader_to_conv0);
        conv0->merge->out(*conv0_to_max1);
        //conv0 = new conv_layer("conv0",0, 416, 416, 3, 3,1,16, 1, LEAKY, true, "conv0.weights");
		//conv0->in(*reader_to_conv0);
		//conv0->out(*conv0_to_max1);

		max1 = new max_layer("max1",1, 416, 416, 16, 2,2);
		max1->in(*conv0_to_max1);
		max1->out(*max1_to_conv2);

		conv2 = new conv_layer("conv2",2, 208, 208, 16, 3,1,32, 1, LEAKY, true, false, NULL, NULL);
		conv2->in(*max1_to_conv2);
		conv2->out(*conv2_to_max3);

		
		max3 = new max_layer("max3",3, 208, 208, 32, 2,2);
        max3->in(*conv2_to_max3);
        max3->out(*max3_to_conv4);
		
    	conv4 = new conv_layer("conv4",4, 104, 104, 32, 3,1,64,1, LEAKY, true, false, NULL, NULL);
        conv4->in(*max3_to_conv4);
        conv4->out(*conv4_to_max5);

		
		max5 = new max_layer("max5",5, 104, 104, 64, 2, 2);
        max5->in(*conv4_to_max5);
        max5->out(*max5_to_conv6);
		
		conv6 = new conv_layer("conv6",6, 52, 52, 64, 3,1,128,1, LEAKY, true, false, NULL, NULL);
        conv6->in(*max5_to_conv6);
        conv6->out(*conv6_to_max7);

	
		max7 = new max_layer("max7",7, 52, 52, 128, 2,2);
        max7->in(*conv6_to_max7);
        max7->out(*max7_to_conv8);
		

		conv8 = new conv_layer("conv8",8, 26, 26, 128, 3,1,256,1, LEAKY ,true, false, NULL, NULL);
        conv8->in(*max7_to_conv8);
        conv8->out(*conv8_to_max9);

		
		max9 = new max_layer("max9",9, 26, 26, 256, 2,2);
        max9->in(*conv8_to_max9);
        max9->out(*max9_to_conv10);
		
		conv10 = new conv_layer("conv10",10, 13, 13, 256, 3,1,512,1, LEAKY, true, false, NULL, NULL);
        conv10->in(*max9_to_conv10);
        conv10->out(*conv10_to_max11);
	
		max11 = new max_layer("max11",11, 13, 13, 512, 2,1);
        max11->in(*conv10_to_max11);
        max11->out(*max11_to_conv12);

		conv12 = new conv_layer("conv12",12, 13, 13, 512, 3,1,1024,1,LEAKY,true, false, NULL, NULL);
        conv12->in(*max11_to_conv12);
        conv12->out(*conv12_to_conv13);
		
		conv13 = new conv_layer("conv13",13, 13, 13, 1024, 3,1,512,1,LEAKY,true, false, NULL, NULL);
        conv13->in(*conv12_to_conv13);
        conv13->out(*conv13_to_conv14);

		conv14 = new conv_layer("conv14",14, 13, 13, 512, 1,1,425,1, LINEAR, false, false, NULL, NULL);
        conv14->in(*conv13_to_conv14);
        conv14->out(*conv14_to_region);
		region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                               true, 1, 1, true, 0.6, true, 13, 13);
		region->in(*conv14_to_region);
		region->im_in(*reader_to_writer);
		region->im_w_in(*int_reader_to_writer); 
		region->im_h_in(*int2_reader_to_writer);
		region->im_name_in(*char_reader_to_writer); 
	}
};

// This will probably remain as-is.
int sc_main(int argc, char * argv[]) 
{
    kpn_neuralnet knn0("kpn_neuralnet");
    sc_start();
    return 0;
}
