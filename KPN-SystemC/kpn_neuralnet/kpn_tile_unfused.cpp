#include <stdio.h>
#include "array_ops.h"
#include <string>
#include "../kahn_process.h"


class   merge_layer : public kahn_process
{
    public:

    const   int *tileWidths;
    const   int *tileHeights;
    const   int numChannels;

    sc_fifo_in<float*> *in;
    sc_fifo_out<float*> out;

    merge_layer(sc_module_name name, int *_tileWidths, int *_tileHeights,  int _numChannels)
    :   kahn_process(name),
        tileWidths(_tileWidths),
        tileHeights(_tileHeights),
        numChannels(_numChannels)
    {
        cout << "instantiated merge layer " << endl;

    }
    
    void    process() override
    {
        float **data = new float*[9];
        for(int i = 0; i < 9; i++){
            in[i]->read(data[i]);   
        }
        cout << "merging tiles @ iter " << iter << endl;
        float *output = mergeTiles(data, this->tileWidths, this->tileHeights, this->numChannels);
        out.write(output);
    }
};

class   scatter_layer : public kahn_process
{
    public:

    const   int **coords;
    const   int width;
    const   int height;
    const   int numChannels;

    sc_fifo_in<float*> in;
    sc_fifo_out<float*> *out;

    scatter_layer(sc_module_name name, const int **_coords, int _width, int _height, int _numChannels)
    :   kahn_process(name),
        coords(_coords),
        width(_width),
        height(_height),
        numChannels(_numChannels)
    {
        cout << "instantiated scatter layer " << endl;

    }

    void    process() override
    {
        float *data;
        in->read(data);
        cout << "scattering tiles @ iter " << iter << endl;
        float **output = new float*[9];
        for(int i = 0; i < 9; i++){
            output[i] = getSubArray(data, this->coords[i], this->width, this->height, this->numChannels);
        }
        for(int i = 0; i < 9; i++){
            out[i].write(output[i]);
        }
         
    }
};




