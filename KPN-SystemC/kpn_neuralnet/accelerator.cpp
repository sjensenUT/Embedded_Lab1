#include "accelerator.h"
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
#include "kpn_neuralnet_fused.h"
#include "kpn_BusSlave.h" 
#include "HWBus.h" 


/*void    load(int lIdx, const char* attr, float* ptr, int size)
{
    char    fn[100];
    FILE*   fh;

    sprintf(fn, "out/l%i/%s.bin", lIdx, attr);
    fh = fopen(fn, "r");
    fread(ptr, sizeof(float), size, fh);
    fclose(fh);
}*/

accelerator::accelerator(sc_module_name name)
:   kahn_process(name)
{
    int filterSize = 3;
    int padding = 1;
    int w1 = 13;
    int w2 = 13;
    int h1 = 13;
    int h2 = 13;
    int c1 = 512;
    int c2 = 1024;
    int batch = 1;
    int numFilters1 = 1024;
    int numFilters2 = 512;
    int groups = 1;
    int stride = 1;
    int batchNormalize = 1;
    int layerIndex1 = 12;
    int layerIndex2 = 13;
    l1 = make_convolutional_layer(batch, h1, w1, c1, numFilters1, groups, filterSize, stride, padding, LEAKY, batchNormalize, 0, 0, 0);
    l2 = make_convolutional_layer(batch, h2, w2, c2, numFilters2, groups, filterSize, stride, padding, LEAKY, batchNormalize, 0, 0, 0);
    
    int num1 = l1.c/l1.groups*l1.n*l1.size*l1.size;
    load(layerIndex1, "biases", l1.biases, l1.n);

    if(l1.batch_normalize)
    {
        load(layerIndex1, "scales", l1.scales, l1.n);
        load(layerIndex1, "mean",   l1.rolling_mean, l1.n);
        load(layerIndex1, "variance", l1.rolling_variance, l1.n);
    }

    load(layerIndex1, "weights", l1.weights, num1);
    
    int num2 = l2.c/l2.groups*l2.n*l2.size*l2.size;
    load(layerIndex2, "biases", l2.biases, l2.n);

    if(l2.batch_normalize)
    {
        load(layerIndex2, "scales", l2.scales, l2.n);
        load(layerIndex2, "mean",   l2.rolling_mean, l2.n);
        load(layerIndex2, "variance", l2.rolling_variance, l2.n);
    }

    load(layerIndex2, "weights", l2.weights, num2);
}

void accelerator::init(){}

void accelerator::process(){
    cout << "in accelerator::process" << endl;    
    float* input;
    input = readImageData(&in, l1.w, l1.h, l1.c);
    
    network dummyNetwork1, dummyNetwork2;
    cout << "allocating network and workspace 1" << endl;
    dummyNetwork1.input = input;
    dummyNetwork1.train = 0;
    size_t workspace_size1 = get_convolutional_workspace_size(l1);
    dummyNetwork1.workspace = (float*) calloc(1, workspace_size1);
    cout << "performing forward convolution 1" << endl;
    forward_convolutional_layer(l1, dummyNetwork1);
    
    cout << "allocating network and workspace 2" << endl;
    dummyNetwork2.train = 0;
    size_t workspace_size2 = get_convolutional_workspace_size(l2);
    //memcpy(dummyNetwork2.input, dummyNetwork1.output, workspace_size2);
    dummyNetwork2.input = l1.output;
    dummyNetwork2.workspace = (float*) calloc(1, workspace_size2);
    cout << "performing forward convolution 2" << endl; 
    forward_convolutional_layer(l2, dummyNetwork2);
    cout << "freeing workspaces" << endl; 
    free(dummyNetwork1.workspace);
    free(dummyNetwork2.workspace);
    
    float* outputImage = l2.output;
    int outputWidth    = l2.out_w;
    int outputHeight   = l2.out_h;
    int outputChans    = 512;
    cout << "writing image data" << endl;
    wait(187,SC_MS); // time of both layers 464 + 448 / 5
    writeImageData(&out, outputImage, outputWidth, outputHeight, outputChans);
}


accelerator_to_bus::accelerator_to_bus(sc_module_name name)
:   kahn_process(name)
{
    int filterSize = 3;
    int padding = 1;
    int w1 = 13;
    int w2 = 13;
    int h1 = 13;
    int h2 = 13;
    int c1 = 512;
    int c2 = 1024;
    int batch = 1;
    int numFilters1 = 1024;
    int numFilters2 = 512;
    int groups = 1;
    int stride = 1;
    int batchNormalize = 1;
    int layerIndex1 = 12;
    int layerIndex2 = 13;
    l1 = make_convolutional_layer(batch, h1, w1, c1, numFilters1, groups, filterSize, stride, padding, LEAKY, batchNormalize, 0, 0, 0);
    l2 = make_convolutional_layer(batch, h2, w2, c2, numFilters2, groups, filterSize, stride, padding, LEAKY, batchNormalize, 0, 0, 0);
    
    int num1 = l1.c/l1.groups*l1.n*l1.size*l1.size;
    load(layerIndex1, "biases", l1.biases, l1.n);

    if(l1.batch_normalize)
    {
        load(layerIndex1, "scales", l1.scales, l1.n);
        load(layerIndex1, "mean",   l1.rolling_mean, l1.n);
        load(layerIndex1, "variance", l1.rolling_variance, l1.n);
    }

    load(layerIndex1, "weights", l1.weights, num1);
    
    int num2 = l2.c/l2.groups*l2.n*l2.size*l2.size;
    load(layerIndex2, "biases", l2.biases, l2.n);

    if(l2.batch_normalize)
    {
        load(layerIndex2, "scales", l2.scales, l2.n);
        load(layerIndex2, "mean",   l2.rolling_mean, l2.n);
        load(layerIndex2, "variance", l2.rolling_variance, l2.n);
    }

    load(layerIndex2, "weights", l2.weights, num2);
}

void accelerator_to_bus::init(){}

void accelerator_to_bus::process(){
    cout << "in accelerator_to_bus::process" << endl;    
    float* input;
    input = new float[l1.w*l1.h*l1.c]; 
//    input = readImageData(&in, l1.w, l1.h, l1.c);
    accel_to_bus->read(input,l1.w*l1.h*l1.c*sizeof(float));     
    cout << "read complete" << endl;  
    cout << "input[0]: " << input[0] << endl;
    network dummyNetwork1, dummyNetwork2;
    cout << "allocating network and workspace 1" << endl;
    dummyNetwork1.input = input;
    dummyNetwork1.train = 0;
    size_t workspace_size1 = get_convolutional_workspace_size(l1);
    dummyNetwork1.workspace = (float*) calloc(1, workspace_size1);
    cout << "performing forward convolution 1" << endl;
    forward_convolutional_layer(l1, dummyNetwork1);
    
    cout << "allocating network and workspace 2" << endl;
    dummyNetwork2.train = 0;
    size_t workspace_size2 = get_convolutional_workspace_size(l2);
    //memcpy(dummyNetwork2.input, dummyNetwork1.output, workspace_size2);
    dummyNetwork2.input = l1.output;
    dummyNetwork2.workspace = (float*) calloc(1, workspace_size2);
    cout << "performing forward convolution 2" << endl; 
    forward_convolutional_layer(l2, dummyNetwork2);
    cout << "freeing workspaces" << endl; 
    free(input);
    free(dummyNetwork1.workspace);
    free(dummyNetwork2.workspace);
    
    float* outputImage = l2.output;
    int outputWidth    = l2.out_w;
    int outputHeight   = l2.out_h;
    int outputChans    = 512;
    cout << "writing image data" << endl;
    wait(187,SC_MS); // time of both layers 464 + 448 / 5

    accel_to_bus->write(outputImage,outputWidth*outputHeight*outputChans*sizeof(float));
         
}
