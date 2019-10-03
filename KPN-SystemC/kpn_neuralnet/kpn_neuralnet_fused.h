class   kpn_fused_thread : public sc_module
{
    public:

    //Declare all queues between our layers here
    //I think the data type for all of them will be image
    sc_fifo<float> *conv0_to_max1,
            *max1_to_conv2,
            *conv2_to_max3,
            *max3_to_conv4,
            *conv4_to_max5,
            *max5_to_conv6,
            *conv6_to_max7,
            *max7_to_conv8,
            *conv8_to_max9,
            *max9_to_conv10,
            *conv10_to_max11,
            *max11_to_conv12,
            *conv12_to_conv13,
            *conv13_to_conv14;

    //Declare all layers here
    max_layer   *max1, *max3, *max5, *max7, *max9, *max11;
    conv_layer  *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;

    //Constructor of the overall network. Initialize all queues and layers
    kpn_fused_thread(sc_module_name name, int *finalOutputCoords);
};
