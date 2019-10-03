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
#include "kpn_neuralnet.h" 
#include "image_data.h"

using	std::cout;
using	std::endl;
using std::string;
using std::size_t;
typedef std::vector<std::string> strs;

// These constants are fixed parameters of the YOLO-V2 Tiny network.
const int IMAGE_WIDTH  = 416;
const int IMAGE_HEIGHT = 416;
const int BATCH        = 1;

const int BIGGEST_FIFO_SIZE = 416 * 416 * 16;

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


image_reader::image_reader(sc_module_name name, strs _images)
:	kahn_process(name),
	images(_images)
{
	cout << "instantiated image_reader" << endl;
}

void image_reader::process()
{
	for(size_t i=0; i<images.size(); i++)
	{
		cout << "reading image " << images[i] << " @ iter " << iter << endl;

		// read images[i] from file
		image orig  = load_image_color( const_cast<char*> (images[i].c_str()), 0, 0);
 		image sized = letterbox_image(orig, IMAGE_WIDTH, IMAGE_HEIGHT);

		// sized.data is now the float* that points to the float array that will
		// be the output/input of each layer. The image writer will call free on 
        // this float* to deallocate the data.
    writeImageData(&out, sized.data, IMAGE_WIDTH, IMAGE_HEIGHT, 3);
		writeImageData(&im_out, orig.data, orig.w, orig.h, 3);
		im_w_out->write(orig.w);
		im_h_out->write(orig.h); //give both width in height in queue of length 2
		char name[10];
        sprintf(name,"image%zu",i);
        string name_str(name);			
        im_name_out->write(name_str);
	}
}


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

void conv_layer::printCoords() {
    printf("Layer %d input coords: %d %d %d %d\n", layerIndex,
        this->inputCoords[0],
        this->inputCoords[1],
        this->inputCoords[2],
        this->inputCoords[3]);  
}

int* getCropCoords (int* inputCoords, int* outputCoords) {
    int* croppedCoords = new int[4];
    int cropped_width = outputCoords[2] - outputCoords[0] + 1;
    int cropped_height = outputCoords[3] - outputCoords[1] + 1;
    int left_crop   = outputCoords[0] - inputCoords[0];
    int top_crop    = outputCoords[1] - inputCoords[1];
    //int right_crop  = inputCoords[2]  - outputCoords[2];
    //int bottom_crop = inputCoords[3]  - outputCoords[3];
    croppedCoords[0] = left_crop;
    croppedCoords[1] = top_crop;
    croppedCoords[2] = left_crop + cropped_width - 1;
    croppedCoords[3] = top_crop + cropped_height - 1;
    return croppedCoords;
}


conv_layer::conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
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
    if (crop) {
        inputCoords = new int[4];
        outputCoords = new int[4];
        for (int j = 0; j < 4; j++) {
            inputCoords[j] = _inputCoords[j];
            outputCoords[j] = _outputCoords[j];
        }
    }

}

void conv_layer::process()
{
    float* input;

    // Read the output from the previos layer
    input = readImageData(&in, l.w, l.h, l.c);
    cout << "forwarding convolutional layer " << layerIndex << " @ iter " << iter << endl;

    // Create a dummy network object. forward_convolutional_layer only uses the "input"
    // and "workspace" elements of the network struct. "input" is simply the output of
    // the previous layer, while "workspace" points to an array of floats that we will
    // create just before calling. The size can be determined by layer.get_workspace_size().
    network dummyNetwork;
    dummyNetwork.input = input;

    //printf("inputs of layer %d, are", layerIndex);
    //for(int j = 0; j < 10; j++){
    //    printf(" %f", input[j]);
    //}
    //printf("\n");

    size_t workspace_size = get_convolutional_workspace_size(l);
    dummyNetwork.workspace = (float*) calloc(1, workspace_size);
    forward_convolutional_layer(l, dummyNetwork);

    //printf("outputs of layer %d, are", layerIndex);
    //for(int j = 0; j < 10; j++){
    //    printf(" %f", l.output[j]);
    //}
    //printf("\n");

    free(dummyNetwork.workspace);

    float* outputImage = l.output;
    int outputWidth    = l.out_w;
    int outputHeight   = l.out_h;
    int outputChans    = this->numFilters;

    // Now it's time to crop the data if this layer is configured to do cropping.
    if (crop) {
    
        // Calculate the relative coordinates for cropping
        int* cropCoords = getCropCoords(inputCoords, outputCoords);
                         
        //printf("Cropping image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
        //        inputCoords[0], inputCoords[1], inputCoords[2], inputCoords[3],
        //        outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
        outputImage  = getSubArray(l.output, cropCoords, l.w, l.h, this->numFilters);
        outputWidth  = cropCoords[2] - cropCoords[0] + 1;
        outputHeight = cropCoords[3] - cropCoords[1] + 1;
    }

    // Send off the layer's output to the next layer!
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans);

}


max_layer::max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
    int _stride, bool _crop, int* _inputCoords, int* _outputCoords)
:	kahn_process(name),
    stride(_stride),
    layerIndex(_layerIndex),
    filterSize(_filterSize),
    crop(_crop)
{
    cout << "instantiated max layer " << layerIndex << " with filter size of " << filterSize << " and stride of " << stride << endl;

    // Create the underlying darknet layer
    l = make_maxpool_layer(BATCH, _h, _w, _c, this->filterSize, 
                       this->stride, filterSize-1);

    if (crop) {
        inputCoords = new int[4];
        outputCoords = new int[4];
        for (int j = 0; j < 4; j++) {
            inputCoords[j] = _inputCoords[j];
            outputCoords[j] = _outputCoords[j];
        }
    }
}


void max_layer::process()
{

    float* data;
    data = readImageData(&in, l.w, l.h, l.c);

    cout << "forwarding max layer " << layerIndex << " @ iter " << iter << endl;

    //printf("inputs of layer %d, are", layerIndex);
    //for(int j = 0; j < 10; j++){
    //  printf(" %f", data[j]);
    //}
    //printf("\n");

    // Call forward_maxpool_layer() here, read from layer.output and write to out
    // Create a dummy network object. The function only uses network.input
    network dummyNetwork;
    dummyNetwork.input = data;
    forward_maxpool_layer(l, dummyNetwork);

    float* outputImage = l.output;
    int outputWidth  = l.out_w;
    int outputHeight = l.out_h;
    int outputChans  = l.c;

    // Now it's time to crop the data if this layer is configured to do cropping.
    if (crop) {
    
        int preCropCoords[4] = { inputCoords[0] / this->stride,
                               inputCoords[1] / this->stride,
                               inputCoords[0] / this->stride + l.out_w - 1,
                               inputCoords[1] / this->stride + l.out_h - 1 };

        // Calculate the relative coordinates for cropping
        int* cropCoords = getCropCoords(preCropCoords, outputCoords);
        //printf("Cropping maxpool image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
        //    preCropCoords[0], preCropCoords[1], preCropCoords[2], preCropCoords[3],
        //    outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
        outputImage  = getSubArray(l.output, cropCoords, l.out_w, l.out_h, l.c);
        outputWidth  = cropCoords[2] - cropCoords[0] + 1;
        outputHeight = cropCoords[3] - cropCoords[1] + 1;
    }
    // Send off the layer's output to the next layer!
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans);	

}


region_layer::region_layer(sc_module_name name, float _anchors[], bool _biasMatch, int _classes,
           int _coords, int _num, bool _softMax, float _jitter, bool _rescore, 
           int _objScale, bool _noObjectScale, int _classScale, int _coordScale,
           bool _absolute, float _thresh, bool _random, int _w, int _h, int _c) 
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
    random(_random),
    chans(_c)
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


void region_layer::process()
{
	float* data;
	string image_name; 
	image im; 
	
  data = readImageData(&in, l.w, l.h, this->chans);

	im_name_in->read(image_name);
	im_w_in->read(im.w);
	im_h_in->read(im.h); 
	im.c = 3;
  im.data = readImageData(&im_in, im.w, im.h, im.c);

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
  free(data);

	cout << "writing predictions to " << outFN << "  @ iter " << iter << endl;
	//free(alphabets);  Now part of the constructor and I don't free it here? 
}

int coerce (int val, int min, int max) {
  if (val < min) return min;
  if (val > max) return max;
  return val;
}

conv_layer_unfused::conv_layer_unfused(sc_module_name name, int layerIndex, int coords[][4],
                   int inputWidth, int inputHeight, int c, int filterSize, int stride, int numFilters, int pad,
                   ACTIVATION activation, bool batchNormalize) : sc_module(name)
{
    cout << "instantiating fused conv layer" << endl;
    cout << "inputWidth = " << inputWidth << ", inputHeight = " << inputHeight << ", c = " << c << endl; 
//    for(int i = 0; i < 9; i++){
//        for(int j = 0; j < 4; j++){
//            cout << "coords[" << i << "][" << j << "] = " << coords[i][j]  << endl;
//        }
//    } 
    int *widths = new int[3]  { coords[0][2] - coords[0][0] + 1,
                                coords[1][2] - coords[1][0] + 1,
                                coords[2][2] - coords[2][0] + 1 };
    int *heights = new int[3] { coords[0][3] - coords[0][1] + 1,
                                coords[3][3] - coords[3][1] + 1,
                                coords[6][3] - coords[6][1] + 1 };

    // Create the padded coordinates
    if(pad){
        int paddedCoords[9][4];
        calcPrevCoords(coords, paddedCoords, stride, filterSize, inputWidth, inputHeight, "convolutional");
        for (int j = 0; j < 9; j++) {
            //printf("Tile %d padded coordinates: (%d, %d) (%d, %d)\n", j,
            //    paddedCoords[j][0], paddedCoords[j][1],
            //    paddedCoords[j][2], paddedCoords[j][3]);
            //cout << "in conv_layer instantiation loop i = " << i << endl;
            int w = paddedCoords[j][2] - paddedCoords[j][0] + 1;
            int h = paddedCoords[j][3] - paddedCoords[j][1] + 1;
            string name = "conv";
            name += "_" + std::to_string(layerIndex) + "_" + std::to_string(j);
            conv[j] = new conv_layer(name.c_str(), layerIndex, w, h, c, filterSize, stride, numFilters, pad, activation, batchNormalize,
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
    merge = new merge_layer("merge", widths, heights, numFilters);
    for(int i = 0; i < 9; i++){
        scatter_to_conv[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
        conv_to_merge[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
        conv[i]->in(*scatter_to_conv[i]);
        conv[i]->out(*conv_to_merge[i]);
        scatter->out[i](*scatter_to_conv[i]);
        merge->in[i](*conv_to_merge[i]);
    }

}

max_layer_unfused::max_layer_unfused(sc_module_name name, int layerIndex, int coords[][4],
                   int inputWidth, int inputHeight, int c, int size, int stride,
                   int pad ) : sc_module(name)
{
    cout << "instantiating fused max layer" << endl;
    cout << "inputWidth = " << inputWidth << ", inputHeight = " << inputHeight << ", c = " << c << endl; 
    //for(int i = 0; i < 9; i++){
    //    for(int j = 0; j < 4; j++){
    //        cout << "(max) coords[" << i << "][" << j << "] = " << coords[i][j]  << endl;
    //    }
    //} 
 
    int *widths = new int[3]  { coords[0][2] - coords[0][0] + 1,
                                coords[1][2] - coords[1][0] + 1,
                                coords[2][2] - coords[2][0] + 1 };
    int *heights = new int[3] { coords[0][3] - coords[0][1] + 1,
                                coords[3][3] - coords[3][1] + 1,
                                coords[6][3] - coords[6][1] + 1 };

    // Create the padded coordinates
    if(pad){
        int paddedCoords[9][4];
        calcPrevCoords(coords, paddedCoords, stride, size, inputWidth, inputHeight, "maxpool");
        for (int j = 0; j < 9; j++) {
            printf("Tile %d padded coordinates (max): (%d, %d) (%d, %d)\n", j,
                paddedCoords[j][0], paddedCoords[j][1],
                paddedCoords[j][2], paddedCoords[j][3]);
            int w = paddedCoords[j][2] - paddedCoords[j][0] + 1;
            int h = paddedCoords[j][3] - paddedCoords[j][1] + 1;

            string name = "max";
            name += "_" + std::to_string(layerIndex) + "_" + std::to_string(j);
            maxl[j] = new max_layer(name.c_str(), layerIndex, w, h, c, size, stride,
                                    true, paddedCoords[j], coords[j]);
        }
        scatter = new scatter_layer("scatter", paddedCoords, inputWidth, inputHeight, c);
    } else {
        for(int j = 0; j < 9; j++){
            int w = coords[j][2] - coords[j][0] + 1;
            int h = coords[j][3] - coords[j][1] + 1;
            maxl[j] = new max_layer("max", layerIndex, w, h, c, size, stride,
                                false, NULL, NULL);
        }
        scatter = new scatter_layer("scatter", coords, inputWidth, inputHeight, c);
    }

    merge = new merge_layer("merge", widths, heights, c);

    for(int i = 0; i < 9; i++){
        scatter_to_max[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
        max_to_merge[i] = new sc_fifo<float>(BIGGEST_FIFO_SIZE);
        maxl[i]->in(*scatter_to_max[i]);
        maxl[i]->out(*max_to_merge[i]);
        scatter->out[i](*scatter_to_max[i]);
        merge->in[i](*max_to_merge[i]);
    }
}



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
			*conv14_to_region;
//			*region_to_writer;

	sc_fifo<float>  *reader_to_writer; 
//	sc_fifo<int>    *layer_region_to_writer; 
	sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	sc_fifo<string>  *char_reader_to_writer; 
			//*conv2_to_detection,
			//*detection_to_writer;

  // Declare all layers here
	conv_layer_unfused *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12,
                     *conv13, *conv14;
  max_layer_unfused *max1, *max3, *max5, *max7, *max9, *max11;
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
		reader_to_conv0 	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);
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
		conv14_to_region   	= new sc_fifo<float>(BIGGEST_FIFO_SIZE);

		reader_to_writer 	= new sc_fifo<float>(800 * 600 * 3); 
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
    int tileCoords[9][4];
    getTileCoords(416, 416, tileCoords);
    conv0 = new conv_layer_unfused("conv0", 0, tileCoords, 416, 416, 3, 3, 1, 16, 1,  LEAKY, true);
    conv0->scatter->in(*reader_to_conv0);
    conv0->merge->out(*conv0_to_max1);

    getTileCoords(208, 208, tileCoords); // These are the output coordinates
    max1 = new max_layer_unfused("max1", 1, tileCoords, 416, 416, 16, 2, 2, true); 
    max1->scatter->in(*conv0_to_max1);
    max1->merge->out(*max1_to_conv2);

    conv2 = new conv_layer_unfused("conv2", 2, tileCoords, 208, 208, 16, 3, 1, 32,
                1, LEAKY, true);
    conv2->scatter->in(*max1_to_conv2);
    conv2->merge->out(*conv2_to_max3);

	  getTileCoords(104, 104, tileCoords); // These are the output coordinates
    max3 = new max_layer_unfused("max3", 3, tileCoords, 208, 208, 32, 2, 2, true); 
    max3->scatter->in(*conv2_to_max3);
    max3->merge->out(*max3_to_conv4);

	  conv4 = new conv_layer_unfused("conv4", 4, tileCoords, 104, 104, 32, 3, 1, 64,
                1, LEAKY, true);
    conv4->scatter->in(*max3_to_conv4);
    conv4->merge->out(*conv4_to_max5);
		
    getTileCoords(52, 52, tileCoords);
		max5 = new max_layer_unfused("max5",5, tileCoords, 104, 104, 64, 2, 2, true);
    max5->scatter->in(*conv4_to_max5);
    max5->merge->out(*max5_to_conv6);
		
    conv6 = new conv_layer_unfused("conv6",6,tileCoords, 52, 52, 64,3,1,128,
                1, LEAKY, true);
    conv6->scatter->in(*max5_to_conv6);
    conv6->merge->out(*conv6_to_max7);

	  getTileCoords(26, 26, tileCoords);
		max7 = new max_layer_unfused("max7",7, tileCoords, 52, 52, 128, 2, 2, true);
    max7->scatter->in(*conv6_to_max7);
    max7->merge->out(*max7_to_conv8);	

		conv8 = new conv_layer_unfused("conv8",8,tileCoords, 26, 26, 128, 3,1,256,
               1, LEAKY ,true);
    conv8->scatter->in(*max7_to_conv8);
    conv8->merge->out(*conv8_to_max9);

	  getTileCoords(13, 13, tileCoords);	
		max9 = new max_layer_unfused("max9",9, tileCoords, 26, 26, 256, 2,2, true);
    max9->scatter->in(*conv8_to_max9);
    max9->merge->out(*max9_to_conv10);
		
		conv10 = new conv_layer_unfused("conv10",10, tileCoords, 13, 13, 256, 3,1,512,
                1, LEAKY, true);
    conv10->scatter->in(*max9_to_conv10);
    conv10->merge->out(*conv10_to_max11);
	
    // !!! NOTE !!! this is the only max layer with stride=1
		max11 = new max_layer_unfused("max11",11, tileCoords, 13, 13, 512, 2, 1, true);
    max11->scatter->in(*conv10_to_max11);
    max11->merge->out(*max11_to_conv12);

		conv12 = new conv_layer_unfused("conv12",12, tileCoords, 13, 13, 512, 3,1,1024,
                1, LEAKY, true);
    conv12->scatter->in(*max11_to_conv12);
    conv12->merge->out(*conv12_to_conv13);
		
		conv13 = new conv_layer_unfused("conv13",13,tileCoords, 13, 13, 1024, 3,1,512,
                1, LEAKY, true);
    conv13->scatter->in(*conv12_to_conv13);
    conv13->merge->out(*conv13_to_conv14);

		conv14 = new conv_layer_unfused("conv14",14,tileCoords, 13, 13, 512, 1,1, 425,
                 1, LINEAR, false);
    conv14->scatter->in(*conv13_to_conv14);
    conv14->merge->out(*conv14_to_region);

		region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                               true, 1, 1, true, 0.6, true, 13, 13, 425);
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
