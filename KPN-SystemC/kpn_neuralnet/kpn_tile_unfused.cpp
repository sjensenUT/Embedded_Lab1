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
            in[i].read(data[i]);   
        }
        cout << "merging tiles "  << " @ iter " << iter << endl;
        float *output = mergeTiles(data, this->tileWidths, this->tileHeights, this->numChannels);
        out.write(output);
    }
};
