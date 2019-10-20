#include <string>
#include "os_channel.h"

void getTileCoords(int width, int height, int coords[9][4]);

class   image_reader : public kahn_process
{
    public:

    std::vector<std::string> images;

    sc_fifo_out<float> out;
    sc_fifo_out<float> im_out;
    sc_fifo_out<int> im_w_out;
    sc_fifo_out<int> im_h_out;
    sc_fifo_out<std::string> im_name_out;
    layer l;
    const int waitTime;
    sc_port<os_channel> os;

    image_reader(sc_module_name name, std::vector<std::string> _images, int _waitTime);
    void    process() override;
    void    init() override;
};

class   conv_layer : public kahn_process
{
    public:

    const   int stride;
    const   int numFilters;
    const   int layerIndex;
    const   int filterSize;
    const   int pad;
    const   ACTIVATION activation;
    const   bool batchNormalize;
    const   bool crop;
    const   int waitTime;
    int* inputCoords;
    int* outputCoords;


    sc_fifo_in<float> in;
    sc_fifo_out<float> out;
    sc_port<os_channel> os;    

    convolutional_layer l;
    
    void printCoords();
    conv_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
             int _stride, int _numFilters, int _pad, ACTIVATION _activation,
             bool _batchNormalize, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime);
    void process() override;
    void init() override;
};


class   max_layer : public kahn_process
{
    public:

    const   int stride;
    const   int layerIndex;
    const   int filterSize;

    sc_fifo_in<float> in;
    sc_fifo_out<float> out;

    layer l;
    const bool crop;
    const int waitTime;
    int* inputCoords;
    int* outputCoords;
    sc_port<os_channel> os;

    max_layer(sc_module_name name, int _layerIndex, int _w, int _h, int _c,  int _filterSize,
            int _stride, bool _crop, int* _inputCoords, int* _outputCoords, int _waitTime);
    void process() override;
    void init() override;
};

class   region_layer : public kahn_process
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
    const int chans;
    const int waitTime;

    sc_fifo_in<float> in;

    sc_fifo_in<float> im_in;
    sc_fifo_in<int> im_w_in; // for width and height of image
    sc_fifo_in<int> im_h_in;
    sc_fifo_in<std::string> im_name_in;
    sc_port<os_channel> os;
    
    image ** alphabets;
    layer l;
    
    region_layer(sc_module_name name, float _anchors[], bool _biasMatch, int _classes,
           int _coords, int _num, bool _softMax, float _jitter, bool _rescore,
           int _objScale, bool _noObjectScale, int _classScale, int _coordScale,
           bool _absolute, float _thresh, bool _random, int _w, int _h, int _c, int _waitTime);
    void    process() override;
    void    init() override;
};


class   conv_layer_unfused : public sc_module
{
    public:
    sc_fifo<float> *scatter_to_conv[9],
        *conv_to_merge[9];

    scatter_layer *scatter;
    conv_layer *conv[9];
    merge_layer *merge;
    conv_layer_unfused(sc_module_name name, int layerIndex, int coords[][4],
                       int inputWidth, int inputHeight, int c, int filterSize, int stride, int numFilters, int pad,
                       ACTIVATION activation, bool batchNormalize);
};

class max_layer_unfused : public sc_module
{
    public:
    sc_fifo<float> *scatter_to_max[9],
        *max_to_merge[9];

    scatter_layer *scatter;
    max_layer *maxl[9];
    merge_layer *merge;
    max_layer_unfused(sc_module_name name, int layerIndex, int coords[][4],
                       int inputWidth, int inputHeight, int c, int size, int stride,
                       int pad );
};
