#include <stdio.h>
#include "merge_scatter.h"
#include "array_ops.h"
#include <string>
#include "../kahn_process.h"
#include "darknet.h"

merge_layer::merge_layer(sc_module_name name, int *_tileWidths, int *_tileHeights,  int _numChannels)
:   kahn_process(name),
    tileWidths(_tileWidths),
    tileHeights(_tileHeights),
    numChannels(_numChannels)
{
    cout << "instantiated merge layer " << endl;

}
   
float get_pixel2(image m, int x, int y, int c)
{
  return m.data[c*m.h*m.w + y*m.w + x];
}
 
void merge_layer::process()
{
        float **data = new float*[9];
        for(int i = 0; i < 9; i++){
            in[i]->read(data[i]);   
        }        
        //cout << "merging tiles @ iter " << iter << endl;
        //cout << "data[0][0] = " << data[0][0] << endl; 
        float *output = mergeTiles(data, this->tileWidths, this->tileHeights, this->numChannels);
        //cout << "finished merging tiles" << endl;

/*
        image outputImage;
        outputImage.w = 416;
        outputImage.h = 416;
        outputImage.c = 16; // hard code for layer 0 output
        outputImage.data = output;

        int x, y, c;
          for (c = 0; c < outputImage.c; c++) {
            printf("Channel %d:\n", c);
            for (y = 0; y < outputImage.h; y++) {
              for (x = 0; x < outputImage.w; x++) {
                printf("%f ", get_pixel2(outputImage, x, y, c));
              }
              printf("\n");
            }
            printf("\n");
          }
          printf("\n");
*/

        out->write(output);
}

scatter_layer::scatter_layer(sc_module_name name, int _coords[][4], int _width, int _height, int _numChannels)
:   kahn_process(name),
    width(_width),
    height(_height),
    numChannels(_numChannels)
{
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 4; j++){
                this->coords[i][j] = _coords[i][j];
            }
        }
        cout << "instantiated scatter layer " << endl;

}

void scatter_layer::process()
{
    float *data;
    in->read(data);
//    cout << "scattering tiles @ iter " << iter << endl;
//    cout << "coords[3] = " << this->coords[3][0] << "," << coords[3][1] << " " << coords[3][2] << "," << coords[3][3] << endl;
    float **output = new float*[9];
    for(int i = 0; i < 9; i++){
//        cout << "in for loop i = " << i << endl;
        output[i] = getSubArray(data, this->coords[i], this->width, this->height, this->numChannels);
    }
    for(int i = 0; i < 9; i++){
        out[i]->write(output[i]);
    }
         
}




