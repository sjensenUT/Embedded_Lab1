#include <stdio.h>
#include "merge_scatter.h"
#include "array_ops.h"
#include <string>
#include "../kahn_process.h"
#include "darknet.h"
#include "image_data.h"

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
 
void merge_layer::init(){}
void merge_layer::process()
{
        int totalWidth = tileWidths[0] + tileWidths[1] + tileWidths[2];
        int totalHeight = tileHeights[0] + tileHeights[1] + tileHeights[2];

        float **data = new float*[9];
        for(int i = 0; i < 9; i++){
            //in[i]->read(data[i]);  
            data[i] = readImageData(&in[i], tileWidths[i%3], tileHeights[i/3], numChannels); 
        }        
        //cout << "merging tiles @ iter " << iter << endl;
        //cout << "data[0][0] = " << data[0][0] << endl; 
        float *output = mergeTiles(data, this->tileWidths, this->tileHeights, this->numChannels);
        //cout << "finished merging tiles" << endl;

/*        image outputImage;
        outputImage.w = 208;
        outputImage.h = 208;
        outputImage.c = 16; // hard code for layer 1 output
        outputImage.data = output;

        int x, y, c;
        if(this->tileWidths[0] < 100) {
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
        }
*/

        //out->write(output);
        writeImageData(&out, output, totalWidth, totalHeight, numChannels);
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

void scatter_layer::init(){}
void scatter_layer::process()
{
    float *data;
    data = readImageData(&in, width, height, numChannels);

//    cout << "scattering tiles @ iter " << iter << endl;
    float **output = new float*[9];
    for(int i = 0; i < 9; i++){
        output[i] = getSubArray(data, this->coords[i], this->width, this->height, this->numChannels);
    }
    for(int i = 0; i < 9; i++){
        //out[i]->write(output[i]);
        int tileWidth  = this->coords[i][2] - this->coords[i][0] + 1;
        int tileHeight = this->coords[i][3] - this->coords[i][1] + 1;
        writeImageData(&out[i], output[i], tileWidth, tileHeight, this->numChannels);
    }
         
}




