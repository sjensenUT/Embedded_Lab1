#include "../kahn_process.h"

class   merge_layer : public kahn_process
{
    public:

    const   int *tileWidths;
    const   int *tileHeights;
    const   int numChannels;
    
    sc_fifo_in<float*> *in;
    sc_fifo_out<float*> out;
    merge_layer(sc_module_name name, int *_tileWidths, int *_tileHeights,  int _numChannels);
    void process() override;

};

class   scatter_layer : public kahn_process
{
    public:

    int coords [9][4];
    const   int width;
    const   int height;
    const   int numChannels;

    sc_fifo_in<float*> in;
    sc_fifo_out<float*> *out;

    scatter_layer(sc_module_name name, int _coords[][4], int _width, int _height, int _numChannels);
    void process() override;
};
