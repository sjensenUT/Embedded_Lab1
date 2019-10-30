#ifndef KPN_NEURALNET_OS_H
#define KPN_NEURALNET_OS_H

#include <string>
#include <systemc.h>
#include "kpn_neuralnet.h"
#include "os_channel.h"
#include "os_sc_fifo.h"
#include "accelerator.h"
using std::string;



class	kpn_neuralnet_os : public sc_module
{
	public:
	
    //Declare all queues between our layers here
    //I think the data type for all of them will be image
	os_sc_fifo<float>	*reader_to_conv0, 
			*conv0_to_max1, 
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
			*conv13_to_conv14,
			*conv14_to_region;

	os_sc_fifo<float>  *reader_to_writer;  
	os_sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	os_sc_fifo<string>  *char_reader_to_writer;
    
    
	conv_layer *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv12, *conv13, *conv14;
    max_layer *max1, *max3, *max5, *max7, *max9, *max11;
    region_layer	*region;
	image_reader	*reader0;
    kpn_neuralnet_os(sc_module_name name);
    os_channel *os; 
    
};


class	kpn_neuralnet_os_accel : public sc_module
{
	public:
	
    //Declare all queues between our layers here
    //I think the data type for all of them will be image
	os_sc_fifo<float>	*reader_to_conv0, 
			*conv0_to_max1, 
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
			//*max11_to_conv12,
			//*conv12_to_conv13,
			//*conv13_to_conv14,
			*conv14_to_region;

	os_sc_fifo<float>  *reader_to_writer;  
	os_sc_fifo<int>    *int_reader_to_writer, *int2_reader_to_writer; 
	os_sc_fifo<string>  *char_reader_to_writer;
    
    
	conv_layer *conv0, *conv2, *conv4, *conv6, *conv8, *conv10, *conv14;
    max_layer *max1, *max3, *max5, *max7, *max9, *max11;
    region_layer	*region;
	image_reader	*reader0;
    kpn_neuralnet_os_accel(sc_module_name name, os_channel *os);
    
};

class	kpn_neuralnet_accelerated : public sc_module
{
	public:
	
    kpn_neuralnet_os_accel *neuralnet;
    accelerator *accel;
    os_channel *os;
    os_to_accel_fifo<float> *os_to_accel;
    accel_to_os_fifo<float> *accel_to_os;

    kpn_neuralnet_accelerated(sc_module_name name);
    
};

#endif // KPN_NEURALNET_OS
