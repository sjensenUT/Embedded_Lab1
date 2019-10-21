#include <systemc.h>
#include "image_data.h"

// Writes image data to a float fifo
// Also frees the data once this is done.
void writeImageData ( sc_fifo_out<float> *out, float* data,
                      int w, int h, int c )
{
    //printf("Attempting to write %d x %d x %d image data to %s.\n", w, h, c, out->name());
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < h; jj++) {
            for (int kk = 0; kk < w; kk++) {
                out->write(data[ii*h*w + jj*w + kk]);
            }
        }
    }
    //printf("Wrote %d x %d x %d image data to %s.\n", w, h, c, out->name());
    //free(data);
}

// Reads image data from a float fifo
float* readImageData ( sc_fifo_in<float> *in,
                     int w, int h, int c ) {
  
    //printf("Attempting to read %d x %d x %d image data from %s.\n", w, h, c, in->name());
    float* data = (float*) calloc(c*h*w, sizeof(float));
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < h; jj++) {
            for (int kk = 0; kk < w; kk++) {
                in->read(data[ii*h*w + jj*w + kk]);
            }
         }
    }
    //printf("Read %d x %d x %d image data from %s.\n", w, h, c, in->name());
    return data;
}
