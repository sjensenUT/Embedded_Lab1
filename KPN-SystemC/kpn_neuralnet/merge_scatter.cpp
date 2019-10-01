#include <stdio.h>
#include "merge_scatter.h"
#include "array_ops.h"
#include <string>
#include "../kahn_process.h"

merge_layer::merge_layer(sc_module_name name, int *_tileWidths, int *_tileHeights,  int _numChannels)
:   kahn_process(name),
    tileWidths(_tileWidths),
    tileHeights(_tileHeights),
    numChannels(_numChannels)
{
    cout << "instantiated merge layer " << endl;

}
    
void merge_layer::process()
{
        float **data = new float*[9];
        for(int i = 0; i < 9; i++){
            in[i]->read(data[i]);   
        }
        cout << "merging tiles @ iter " << iter << endl;
        float *output = mergeTiles(data, this->tileWidths, this->tileHeights, this->numChannels);
        out.write(output);
}

scatter_layer::scatter_layer(sc_module_name name, int **_coords, int _width, int _height, int _numChannels)
:   kahn_process(name),
    coords(_coords),
    width(_width),
    height(_height),
    numChannels(_numChannels)
{
        cout << "instantiated scatter layer " << endl;

}

void scatter_layer::process()
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




