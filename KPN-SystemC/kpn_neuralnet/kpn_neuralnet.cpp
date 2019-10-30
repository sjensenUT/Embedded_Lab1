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
#include <chrono>
#include "kpn_neuralnet.h"
#include "kpn_neuralnet_os.h"
#include "kpn_neuralnet_os_bus.h"
#include "kpn_BusSlave.h"
#include "kpn_BusMaster.h"
 
using	std::cout;
using	std::endl;
using std::string;
using std::size_t;
using std::chrono::system_clock; 
using std::chrono::milliseconds; 
typedef std::vector<std::string> strs;

// this should match the number of iterations in the kahn process.h file
const int ITER_MAX = 2;
#define OS_ENABLE TRUE

// These constants are fixed parameters of the YOLO-V2 Tiny network.
const int IMAGE_WIDTH  = 416;
const int IMAGE_HEIGHT = 416;
const int BATCH        = 1;

const int BIGGEST_FIFO_SIZE = 1;

// Constant float array to hold the anchors for the region layer
// whatever that means
const float ANCHORS[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                           5.47434, 7.88282 , 3.52778, 9.77052, 9.16828};


//deprecated once accelerator used
const int LATENCY[17] = {30, 178, 12, 218, 7, 147, 2, 118, 1, 106, 1, 119, 1, 464, 448, 20, 4};
//const int CONV_LATENCY[9] = {178,218,147,118,106,119,464,448,20}; 
//const int MAXP_LATENCY[6] = {12,7,2,1,1,1}; 
const int ITER_SIZE = 30; 
sc_core::sc_time ITER_TIME[ITER_SIZE];

int latencyIndex = 0; 


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


image_reader::image_reader(sc_module_name name, strs _images, int _waitTime)
:	kahn_process(name),
	images(_images),
    waitTime(_waitTime)
{
	cout << "instantiated image_reader" << endl;
}

void image_reader::init(){
    cout << "in image_reader::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task" << endl;
        this->os->reg_task(this->name());
    }
}

void image_reader::process()
{
	//wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++; 
#if OS_ENABLE == TRUE
    int iter = 0;
    while(true){
#endif 
    cout << "top of image reader @ iter " << iter << endl;
    for(size_t i=0; i<images.size(); i++)
	{
		
        cout << "reading image " << images[i] << " @ iter " << iter << endl;

        ITER_TIME[iter%ITER_SIZE] = sc_time_stamp();
        cout << "iter_time: " << ITER_TIME[iter%ITER_SIZE] << " @ iter " << iter << endl;

		// read images[i] from file
		image orig  = load_image_color( const_cast<char*> (images[i].c_str()), 0, 0);
 		image sized = letterbox_image(orig, IMAGE_WIDTH, IMAGE_HEIGHT);
        cout << "read image" << endl;
		// sized.data is now the float* that points to the float array that will
		// be the output/input of each layer. The image writer will call free on 
        // this float* to deallocate the data.
        //int layer_waitTime = LATENCY[latencyIndex];
        //latencyIndex++; latencyIndex %= 17;
        if(this->waitTime > 0){
            this->os->time_wait(30); // hard-coded wait time for the region layer
        }
        writeImageData(&out, sized.data, IMAGE_WIDTH, IMAGE_HEIGHT, 3 );
		writeImageData(&im_out, orig.data, orig.w, orig.h, 3 );
		im_w_out->write(orig.w);
		im_h_out->write(orig.h); //give both width in height in queue of length 2
		char name[10];
        sprintf(name,"image%zu",i);
        string name_str(name);			
        im_name_out->write(name_str); 
       
	}

    cout << "finished reading" << endl; 
    if(this->waitTime > 0)
    {
        //yielding so other tasks can run
        if(iter >= ITER_MAX-1)
        {   
            cout << "terminating image reader @ iter " << iter << endl;     
            this->os->task_terminate();
            break; // exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //cout << "terminating" << endl; 
    }

#if OS_ENABLE == TRUE
    iter++; 
    } // while(true)
#endif 
    
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
         bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime)
:	kahn_process(name),
    stride(_stride),
    numFilters(_numFilters),
    layerIndex(_layerIndex),
    filterSize(_filterSize),
    pad(_pad),
    activation(_activation),
    batchNormalize(_batchNormalize),
    crop(_crop),
    waitTime(_waitTime)
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

void conv_layer::init(){
    cout << "in conv_layer::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task" << endl;
        this->os->reg_task(this->name());
    } 
    //cout << "asd wait time is:" << this->waitTime << endl; 
}

void conv_layer::process()
{

#if OS_ENABLE == TRUE
    int iter = 0; 
    while(true){
#endif 
    
    float* input;
    // Read the output from the previos layer
    input = readImageData(&in, l.w, l.h, l.c);
	
    //wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++;
    cout << "forwarding convolutional layer " << layerIndex << " @ iter " << iter << endl;
    
    // Create a dummy network object. forward_convolutional_layer only uses the "input"
    // and "workspace" elements of the network struct. "input" is simply the output of
    // the previous layer, while "workspace" points to an array of floats that we will
    // create just before calling. The size can be determined by layer.get_workspace_size().
    network dummyNetwork;
    dummyNetwork.input = input;
    dummyNetwork.train = 0; 
    //cout << "Hello1" << endl;
    //printf("inputs of layer %d, are", layerIndex);
    //for(int j = 0; j < 10; j++){
    //    printf(" %f", input[j]);
    //}
    //printf("\n");i
    system_clock::time_point before = system_clock::now(); 
    size_t workspace_size = get_convolutional_workspace_size(l);
    dummyNetwork.workspace = (float*) calloc(1, workspace_size);
    //cout << "performing forward convolution" << endl;
    //cout << "l.outputs = " << l.outputs << endl;
    //cout << "l.batch = " << l.batch << endl;
    //cout << "l.output[0] = " << l.output[0] << endl;
    forward_convolutional_layer(l, dummyNetwork);
    //cout << "l.output[0] = " << l.output[0] << endl; 
    printf("outputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", l.output[j]);
    }
    printf("\n"); 
    unsigned long memoryFootprint = (l.nweights * sizeof(float) + l.inputs * sizeof(float) + workspace_size + l.outputs * sizeof(float))/1024; 
    //cout << "Hello4" << endl;


    free(dummyNetwork.workspace);

    float* outputImage = l.output;
    int outputWidth    = l.out_w;
    int outputHeight   = l.out_h;
    int outputChans    = this->numFilters;
    //cout << "Hello5" << endl;
    // Now it's time to crop the data if this layer is configured to do cropping.
    if (crop) {
        //cout << "Hello6" << endl;
        // Calculate the relative coordinates for cropping
        int* cropCoords = getCropCoords(inputCoords, outputCoords);
                         
        //printf("Cropping image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
        //        inputCoords[0], inputCoords[1], inputCoords[2], inputCoords[3],
        //        outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
        outputImage  = getSubArray(l.output, cropCoords, l.w, l.h, this->numFilters);
        outputWidth  = cropCoords[2] - cropCoords[0] + 1;
        outputHeight = cropCoords[3] - cropCoords[1] + 1;
    }
    
    system_clock::time_point after = system_clock::now(); 
    //cout << "Hello7" << endl;
    milliseconds duration = std::chrono::duration_cast<milliseconds> (after - before); 
    
    cout << "conv layer " << layerIndex << " data: Memory(kB): " << memoryFootprint << " time(ms): " << duration.count() << endl;   
    // Send off the layer's output to the next layer!
    
    int layer_waitTime = LATENCY[layerIndex+1];
//    latencyIndex++; latencyIndex %= 17;
    if(this->waitTime > 0){
        this->os->time_wait(layer_waitTime);
    }
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans);
    
    if(this->waitTime > 0)
    {
        //yielding so other tasks can run
        if(iter >= ITER_MAX-1)
        {        
            cout << "terminating conv layer " << layerIndex << " @ iter " << iter << endl;
            this->os->task_terminate();
            break;// exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //this->os->task_terminate(); 
    }
    
#if OS_ENABLE == TRUE
    iter++; 
    } // while(true)
#endif 
}


max_layer::max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
    int _stride, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime)
:	kahn_process(name),
    stride(_stride),
    layerIndex(_layerIndex),
    filterSize(_filterSize),
    crop(_crop),
    waitTime(_waitTime)
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

void max_layer::init(){
    cout << "in max_layer::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task";
        this->os->reg_task(this->name());
    }
}

void max_layer::process()
{
#if OS_ENABLE == TRUE
    int iter = 0; 
    while(true){
#endif 
    
    float* data;
    data = readImageData(&in, l.w, l.h, l.c );

    //wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++;
    cout << "forwarding max layer " << layerIndex << " @ iter " << iter << endl;

    printf("inputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
      printf(" %f", data[j]);
    }
    printf("\n");

    // Call forward_maxpool_layer() here, read from layer.output and write to out
    // Create a dummy network object. The function only uses network.input
    system_clock::time_point before = system_clock::now(); 
     
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
    
    
    system_clock::time_point after = system_clock::now();
    milliseconds duration = std::chrono::duration_cast<milliseconds> (after-before);

    unsigned long memoryFootprint = ((l.inputs+l.outputs)*sizeof(float))/1024;
    //cout << "conv layer " << layerIndex << " data: Memory(kB): " << memoryFootprint << " time(ms): " << duration.count() << endl;   
    // Send off the layer's output to the next layer!

    int layer_waitTime = LATENCY[layerIndex+1];
//    latencyIndex++; latencyIndex %= 17;
    if(this->waitTime > 0){
        this->os->time_wait(layer_waitTime);
    }
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans );	

    if(this->waitTime > 0)
    {
        //yielding so other tasks can run
       
        if(iter >= ITER_MAX-1)
        {       
            cout << "terminating max layer " << layerIndex << " @ iter " << iter << endl; 
            this->os->task_terminate();
            break; // exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //this->os->task_terminate(); 
    }
#if OS_ENABLE == TRUE   
    iter++;
    } // while(true)
#endif 
}


region_layer::region_layer(sc_module_name name, float _anchors[], bool _biasMatch, int _classes,
           int _coords, int _num, bool _softMax, float _jitter, bool _rescore, 
           int _objScale, bool _noObjectScale, int _classScale, int _coordScale,
           bool _absolute, float _thresh, bool _random, int _w, int _h, int _c, int _waitTime) 
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
    chans(_c),
    waitTime(_waitTime)
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

void region_layer::init(){
    cout << "in region_layer::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task" << endl;
        this->os->reg_task(this->name());
    }
}

void region_layer::process()
{
#if OS_ENABLE == TRUE
    int iter;   
    while(true){
#endif 

	float* data;
	string image_name; 
	image im; 
	
    data = readImageData(&in, l.w, l.h, this->chans );

	im_name_in->read(image_name);
	im_w_in->read(im.w);
	im_h_in->read(im.h); 
	im.c = 3;
    im.data = readImageData(&im_in, im.w, im.h, im.c );

    //wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++;
	cout << "forwarding detection layer @ iter " << iter << endl;
 
    network dummyNetwork;
	dummyNetwork.input = data;
    cout << "calling forward_region_layer" << endl;
    dummyNetwork.train = 0;
    cout << "dummyNetwork.train = " << dummyNetwork.train;
	forward_region_layer(l, dummyNetwork);
    cout << "finished forward_region_layer" << endl;
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

    ITER_TIME[iter%ITER_SIZE] = sc_time_stamp() + sc_time(4,SC_MS) - ITER_TIME[iter%ITER_SIZE];
    cout << "ITER_TIME: " << ITER_TIME[iter%ITER_SIZE] << " @ iter " << endl;
    int layer_waitTime = 4; // hard coded value from measurements
//    latencyIndex++; latencyIndex %= 17;
    if(this->waitTime > 0){
        this->os->time_wait(layer_waitTime);
    }
	cout << "writing predictions to " << outFN << "  @ iter " << iter << endl;	
    cout << "TIMESTAMP: " << sc_time_stamp() << endl << endl; 
    
    if(this->waitTime >0){
        if(iter >= ITER_MAX-1)
        {        
            cout << "terminating region layer @ iter "<< iter << endl; 
            this->os->task_terminate();
            break; // exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //this->os->task_terminate(); 
    }
    //free(alphabets);  Now part of the constructor and I don't free it here? 
    
#if OS_ENABLE == TRUE
    cout << "Incrementing iteration" << endl;
    iter ++;     
    } // while(true)    
#endif 

}


idle_task::idle_task(sc_module_name name,int _waitTime)
    : kahn_process(name), waitTime(_waitTime)
{
    cout << "instantiating idle task" << endl;
}
void idle_task::init()
{
    cout << "in idle_task::init()" << endl;
    if(this->waitTime > 0){
        char name[] = "idle task"; 
        cout << "detected os, registering task" << endl;
        this->os->reg_task(name);
    }
}

void idle_task::process()
{
    while(1)
    {
        if(this->waitTime > 0)    
        {
            cout << "in idle task\n";  
            this->os->time_wait(this->waitTime);
        }
    }
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
                true, paddedCoords[j], coords[j], -1);
        }
        scatter = new scatter_layer("scatter", paddedCoords, inputWidth, inputHeight, c);

    }else{
        for(int j = 0; j < 9; j++){
            //cout << "in conv_layer instantiation loop i = " << i << endl;
            int w = coords[j][2] - coords[j][0] + 1;
            int h = coords[j][3] - coords[j][1] + 1;
            conv[j] = new conv_layer("conv", layerIndex, w, h, c, filterSize, stride, numFilters, pad, activation, batchNormalize,
                                  false, NULL, NULL, -1);
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
    //        cout << "(max) coords[" << nt *heights = new int[3] { coords[0][3] - coords[0][1] + 1,
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
                     true, paddedCoords[j], coords[j], -1);
        }
    } else {
        for(int j = 0; j < 9; j++){
            int w = coords[j][2] - coords[j][0] + 1;
            int h = coords[j][3] - coords[j][1] + 1;
            maxl[j] = new max_layer("max", layerIndex, w, h, c, size, stride,
                                false, NULL, NULL, -1);
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
	
    //Declare all queues between our layers here
    //I think the data type for all of them will be image
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

	sc_fifo<float>  *reader_to_writer;  
	sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	sc_fifo<string>  *char_reader_to_writer; 

    // Declare all layers here
	conv_layer *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12,
                     *conv13, *conv14;
    max_layer *max1, *max3, *max5, *max7, *max9, *max11;
    region_layer	*region;
	image_reader	*reader0;

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
		
        os_channel *os = new os_channel("dummy_os", 100, false);
        
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
	    reader0 = new image_reader("image_reader", images, -1);
		reader0->out(*reader_to_conv0);
		reader0->im_out(*reader_to_writer);
		reader0->im_w_out(*int_reader_to_writer); 
		reader0->im_h_out(*int2_reader_to_writer);
		reader0->im_name_out(*char_reader_to_writer);
        reader0->os(*os);

		//name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
        conv0 = new conv_layer("conv0", 0, 416, 416, 3, 3, 1, 16, 1,  LEAKY, true, false, NULL, NULL, -1);
        conv0->in(*reader_to_conv0);
        conv0->out(*conv0_to_max1);
        conv0->os(*os);

        max1 = new max_layer("max1", 1, 416, 416, 16, 2, 2, false, NULL, NULL, -1); 
        max1->in(*conv0_to_max1);
        max1->out(*max1_to_conv2);
        max1->os(*os);

        conv2 = new conv_layer("conv2", 2, 208, 208, 16, 3, 1, 32, 1, LEAKY, true, false, NULL, NULL, -1);
        conv2->in(*max1_to_conv2);
        conv2->out(*conv2_to_max3);
        conv2->os(*os);

        max3 = new max_layer("max3", 3, 208, 208, 32, 2, 2, false, NULL, NULL, -1); 
        max3->in(*conv2_to_max3);
        max3->out(*max3_to_conv4);
        max3->os(*os);

	    conv4 = new conv_layer("conv4", 4, 104, 104, 32, 3, 1, 64, 1, LEAKY, true, false, NULL, NULL, -1);
        conv4->in(*max3_to_conv4);
        conv4->out(*conv4_to_max5);
        conv4->os(*os);		

		max5 = new max_layer("max5", 5, 104, 104, 64, 2, 2, false, NULL, NULL, -1);
        max5->in(*conv4_to_max5);
        max5->out(*max5_to_conv6);
		max5->os(*os);
        
        conv6 = new conv_layer("conv6", 6, 52, 52, 64, 3, 1, 128, 1, LEAKY, true, false, NULL, NULL, -1);
        conv6->in(*max5_to_conv6);
        conv6->out(*conv6_to_max7);
        conv6->os(*os);

		max7 = new max_layer("max7", 7, 52, 52, 128, 2, 2, false, NULL, NULL, -1);
        max7->in(*conv6_to_max7);
        max7->out(*max7_to_conv8);	
        max7->os(*os);

		conv8 = new conv_layer("conv8", 8, 26, 26, 128, 3, 1, 256, 1, LEAKY, true, false, NULL, NULL, -1);
        conv8->in(*max7_to_conv8);
        conv8->out(*conv8_to_max9);
        conv8->os(*os);

		max9 = new max_layer("max9", 9, 26, 26, 256, 2,2, false, NULL, NULL, -1);
        max9->in(*conv8_to_max9);
        max9->out(*max9_to_conv10);
        max9->os(*os);		

		conv10 = new conv_layer("conv10", 10, 13, 13, 256, 3, 1, 512, 1, LEAKY, true, false, NULL, NULL, -1);
        conv10->in(*max9_to_conv10);
        conv10->out(*conv10_to_max11);
	    conv10->os(*os);
        
        // !!! NOTE !!! this is the only max layer with stride=1
		max11 = new max_layer("max11", 11, 13, 13, 512, 2, 1, false, NULL, NULL, -1);
        max11->in(*conv10_to_max11);
        max11->out(*max11_to_conv12);
        max11->os(*os);

		conv12 = new conv_layer("conv12", 12, 13, 13, 512, 3, 1, 1024, 1, LEAKY, true, false, NULL, NULL, -1);
        conv12->in(*max11_to_conv12);
        conv12->out(*conv12_to_conv13);
		conv12->os(*os);
        
		conv13 = new conv_layer("conv13", 13, 13, 13, 1024, 3, 1, 512, 1, LEAKY, true, false, NULL, NULL, -1);
        conv13->in(*conv12_to_conv13);
        conv13->out(*conv13_to_conv14);
        conv13->os(*os);

		conv14 = new conv_layer("conv14", 14, 13, 13, 512, 1, 1, 425, 1, LINEAR, false, false, NULL, NULL, -1);
        conv14->in(*conv13_to_conv14);
        conv14->out(*conv14_to_region);
        conv14->os(*os);

		region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                               true, 1, 1, true, 0.6, true, 13, 13, 425, -1);
		region->in(*conv14_to_region);
		region->im_in(*reader_to_writer);
		region->im_w_in(*int_reader_to_writer); 
		region->im_h_in(*int2_reader_to_writer);
		region->im_name_in(*char_reader_to_writer);
        region->os(*os);
	}
};


conv_layer_to_bus::conv_layer_to_bus(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
         int _stride, int _numFilters, int _pad, ACTIVATION _activation,
         bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime)
:	kahn_process(name),
    stride(_stride),
    numFilters(_numFilters),
    layerIndex(_layerIndex),
    filterSize(_filterSize),
    pad(_pad),
    activation(_activation),
    batchNormalize(_batchNormalize),
    crop(_crop),
    waitTime(_waitTime)
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

void conv_layer_to_bus::init(){
    cout << "in conv_layer_to_bus::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task" << endl;
        this->os->reg_task(this->name());
    } 
}

void conv_layer_to_bus::process()
{
#if OS_ENABLE == TRUE
    int iter = 0;  
    while(true){
#endif 
    
    float* input;
    input = new float[l.w*l.h*l.c];
    
    cout << "maxlayer giving master read" << endl;
    // Read the output from the previos layer
    mDriver->read(input,l.w*l.h*l.c*sizeof(float));
   
    cout << "finished reading" << endl;  
    //wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++;
    cout << "forwarding convolutional layer " << layerIndex << " @ iter " << iter << endl;
    
    // Create a dummy network object. forward_convolutional_layer only uses the "input"
    // and "workspace" elements of the network struct. "input" is simply the output of
    // the previous layer, while "workspace" points to an array of floats that we will
    // create just before calling. The size can be determined by layer.get_workspace_size().
    network dummyNetwork;
    dummyNetwork.input = input;
    dummyNetwork.train = 0; 
    //cout << "Hello1" << endl;
    //printf("inputs of layer %d, are", layerIndex);
    //for(int j = 0; j < 10; j++){
    //    printf(" %f", input[j]);
    //}
    //printf("\n");i
    system_clock::time_point before = system_clock::now(); 
    size_t workspace_size = get_convolutional_workspace_size(l);
    dummyNetwork.workspace = (float*) calloc(1, workspace_size);
    //cout << "performing forward convolution" << endl;
    //cout << "l.outputs = " << l.outputs << endl;
    //cout << "l.batch = " << l.batch << endl;
    //cout << "l.output[0] = " << l.output[0] << endl;
    forward_convolutional_layer(l, dummyNetwork);
    //cout << "l.output[0] = " << l.output[0] << endl; 
    printf("outputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
        printf(" %f", l.output[j]);
    }
    printf("\n"); 
    unsigned long memoryFootprint = (l.nweights * sizeof(float) + l.inputs * sizeof(float) + workspace_size + l.outputs * sizeof(float))/1024; 
    //cout << "Hello4" << endl;

    free(input); 
    free(dummyNetwork.workspace);

    float* outputImage = l.output;
    int outputWidth    = l.out_w;
    int outputHeight   = l.out_h;
    int outputChans    = this->numFilters;
    //cout << "Hello5" << endl;
    // Now it's time to crop the data if this layer is configured to do cropping.
    if (crop) {
        //cout << "Hello6" << endl;
        // Calculate the relative coordinates for cropping
        int* cropCoords = getCropCoords(inputCoords, outputCoords);
                         
        //printf("Cropping image from (%d, %d) (%d, %d) to (%d, %d) (%d, %d)\n",
        //        inputCoords[0], inputCoords[1], inputCoords[2], inputCoords[3],
        //        outputCoords[0], outputCoords[1], outputCoords[2], outputCoords[3]);
        outputImage  = getSubArray(l.output, cropCoords, l.w, l.h, this->numFilters);
        outputWidth  = cropCoords[2] - cropCoords[0] + 1;
        outputHeight = cropCoords[3] - cropCoords[1] + 1;
    }
    
    system_clock::time_point after = system_clock::now(); 
    //cout << "Hello7" << endl;
    milliseconds duration = std::chrono::duration_cast<milliseconds> (after - before); 
    
    cout << "conv layer " << layerIndex << " data: Memory(kB): " << memoryFootprint << " time(ms): " << duration.count() << endl;   
    // Send off the layer's output to the next layer!
    
    int layer_waitTime = LATENCY[layerIndex+1];
//    latencyIndex++; latencyIndex %= 17;
    if(this->waitTime > 0){
        this->os->time_wait(layer_waitTime);
    }
   
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans);
    
    if(this->waitTime > 0)
    {
        //yielding so other tasks can run
        if(iter >= ITER_MAX-1)
        {       
            cout << "terminating conv layer " << layerIndex << " @ iter " << iter <<  endl; 
            this->os->task_terminate();
            break; // exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //this->os->task_terminate(); 
    }
#if OS_ENABLE == TRUE
    iter++; 
    } // while(true)
#endif 
}

max_layer_to_bus::max_layer_to_bus(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
    int _stride, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime)
:	kahn_process(name),
    stride(_stride),
    layerIndex(_layerIndex),
    filterSize(_filterSize),
    crop(_crop),
    waitTime(_waitTime)
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

void max_layer_to_bus::init(){
    cout << "in max_layer::init()" << endl;
    if(this->waitTime > 0){
        cout << "detected os, registering task";
        this->os->reg_task(this->name());
    }
}

void max_layer_to_bus::process()
{
#if OS_ENABLE == TRUE
    int iter = 0; 
    while(true){
#endif 
    
    float* data;
    data = readImageData(&in, l.w, l.h, l.c );

    //wait(LATENCY[latencyIndex],SC_MS);
    //cout << "waited for " << LATENCY[latencyIndex] << endl;
    //latencyIndex++;
    cout << "forwarding max layer " << layerIndex << " @ iter " << iter << endl;

    printf("inputs of layer %d, are", layerIndex);
    for(int j = 0; j < 10; j++){
      printf(" %f", data[j]);
    }
    printf("\n");

    // Call forward_maxpool_layer() here, read from layer.output and write to out
    // Create a dummy network object. The function only uses network.input
    system_clock::time_point before = system_clock::now(); 
     
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
    
    
    system_clock::time_point after = system_clock::now();
    milliseconds duration = std::chrono::duration_cast<milliseconds> (after-before);

    unsigned long memoryFootprint = ((l.inputs+l.outputs)*sizeof(float))/1024;
    //cout << "conv layer " << layerIndex << " data: Memory(kB): " << memoryFootprint << " time(ms): " << duration.count() << endl;   
    // Send off the layer's output to the next layer!

    int layer_waitTime = LATENCY[layerIndex+1];
//    latencyIndex++; latencyIndex %= 17;
    if(this->waitTime > 0){
        this->os->time_wait(layer_waitTime);
    }
//    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans );	
    cout << "Maxlayer finished. Attempting to write to the bus" << endl; 
    cout << "Output[0] " << outputImage[0] << endl;
    mDriver->write(outputImage,outputWidth*outputHeight*outputChans*sizeof(float)); 

    if(this->waitTime > 0)
    {
        //yielding so other tasks can run
        if(iter >= ITER_MAX-1)
        {        
            cout << "terminating max layer " <<  layerIndex << " @ iter " << iter << endl;
            this->os->task_terminate();
            break;// exit the while loop
        } else {
            this->os->time_wait(0);
        }
        //this->os->task_terminate(); 
    }
#if OS_ENABLE == TRUE
    iter++; 
    } // while(true)
#endif 
    
}

// This will probably remain as-is.
int sc_main(int argc, char * argv[]) 
{
    kpn_neuralnet knn0("kpn_neuralnet");
    //kpn_neuralnet_fused knn0("kpn_neuralnet_fused");
    //kpn_neuralnet_os knn0("kpn_neuralnet_os");
    //kpn_neuralnet_accelerated knn0("kpn_neuralnet_accelerated");
    //kpn_neuralnet_accelerated_bus knn0("kpn_neuralnet_accelerated_bus", false);
    sc_start();
    return 0;
}
