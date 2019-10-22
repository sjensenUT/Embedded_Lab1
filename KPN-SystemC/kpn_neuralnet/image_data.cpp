#include <systemc.h>
#include "image_data.h"

// Writes image data to a float fifo
// Also frees the data once this is done.
void writeImageData ( sc_fifo_out<float> *out, float* data,
                      int w, int h, int c )
{
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < h; jj++) {
            for (int kk = 0; kk < w; kk++) {
                out->write(data[ii*h*w + jj*w + kk]);
            }
        }
    }
    //free(data);
}

// Reads image data from a float fifo
float* readImageData ( sc_fifo_in<float> *in,
                     int w, int h, int c ) {
    float* data = (float*) calloc(c*h*w, sizeof(float));
    for (int ii = 0; ii < c; ii++) {
        for (int jj = 0; jj < h; jj++) {
            for (int kk = 0; kk < w; kk++) {
                in->read(data[ii*h*w + jj*w + kk]);
            }
         }
    }
    return data;
}
