class   merge_layer : public kahn_process
{
    public:

    const   int *tileWidths;
    const   int *tileHeights;
    const   int numChannels;
    
    sc_fifo_in<float*> *in;
    sc_fifo_out<float*> out;
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
};
