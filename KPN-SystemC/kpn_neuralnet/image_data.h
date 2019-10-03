#include <systemc.h>

// Writes image data to a float fifo
// Also frees the data once this is done.
void writeImageData ( sc_fifo_out<float> *out, float* data,
                      int w, int h, int c );

// Reads image data from a float fifo
float* readImageData ( sc_fifo_in<float> *in,
                     int w, int h, int c );
