#include "kpn_neuralnet_os.h"
#include <string>

using   std::cout;
using   std::endl;
using std::string;
typedef std::vector<std::string> strs;
const int BIGGEST_FIFO_SIZE = 1;

const float ANCHORS[10] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                           5.47434, 7.88282 , 3.52778, 9.77052, 9.16828};


kpn_neuralnet_os::kpn_neuralnet_os(sc_module_name name) : sc_module(name)
{
    cout << "in kpn_neuralnet_os" << endl;
    //strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
    strs images = {"../../darknet/data/dog.jpg"};
    //std::string cfgFile = "../../darknet/cfg/yolov2-tiny.cfg";
    //std::string weightFile = "../../darknet/yolov2-tiny.weights";
    //char *cfgFileC = new char[cfgFile.length() + 1];
    //strcpy(cfgFileC, cfgFile.c_str());
    //char *weightFileC = new char[weightFile.length() + 1];
    //strcpy(weightFileC, weightFile.c_str());
    //network *net = load_network(cfgFileC, weightFileC, 0);
    cout << "instantiating os" << endl;
    bool verbose = false;
    os = new os_channel("os", 100, verbose);
    cout << "instantiating channels" << endl;
    reader_to_conv0 	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    reader_to_conv0->os(*os);
    conv0_to_max1   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv0_to_max1->os(*os);
    max1_to_conv2   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max1_to_conv2->os(*os);
    conv2_to_max3   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv2_to_max3->os(*os);
    max3_to_conv4   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max3_to_conv4->os(*os);
    conv4_to_max5   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv4_to_max5->os(*os);
    max5_to_conv6   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max5_to_conv6->os(*os);
    conv6_to_max7   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv6_to_max7->os(*os);
    max7_to_conv8   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max7_to_conv8->os(*os);
    conv8_to_max9   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv8_to_max9->os(*os);
    max9_to_conv10   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max9_to_conv10->os(*os);
    conv10_to_max11   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv10_to_max11->os(*os);
    max11_to_conv12    = new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max11_to_conv12->os(*os);
    conv12_to_conv13    = new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv12_to_conv13->os(*os); 
    conv13_to_conv14    = new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv13_to_conv14->os(*os); 
    conv14_to_region   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv14_to_region->os(*os);

    reader_to_writer 	= new os_sc_fifo<float>(800 * 600 * 3 * 20);
    reader_to_writer->os(*os);
    int_reader_to_writer	= new os_sc_fifo<int>(20); // needed to send im.w and im.h
    int_reader_to_writer->os(*os);
    int2_reader_to_writer 	= new os_sc_fifo<int>(20);
    int2_reader_to_writer->os(*os); 
    char_reader_to_writer  	= new os_sc_fifo<string>(20);
    char_reader_to_writer->os(*os);
    
    // Here is where we will indicate the parameters for each layer. These can
    // be found in the cfg file for yolov2-tiny in the darknet folder.
    reader0 = new image_reader("image_reader", images, 1);
    reader0->out(*reader_to_conv0);
    reader0->im_out(*reader_to_writer);
    reader0->im_w_out(*int_reader_to_writer); 
    reader0->im_h_out(*int2_reader_to_writer);
    reader0->im_name_out(*char_reader_to_writer);
    reader0->os(*os);


    //name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
    conv0 = new conv_layer("conv0", 0, 416, 416, 3, 3, 1, 16, 1,  LEAKY, true, false, NULL, NULL, 1);
    conv0->in(*reader_to_conv0);
    conv0->out(*conv0_to_max1);
    conv0->os(*os);
    
    max1 = new max_layer("max1", 1, 416, 416, 16, 2, 2, false, NULL, NULL, 1); 
    max1->in(*conv0_to_max1);
    max1->out(*max1_to_conv2);
    max1->os(*os);
    
    conv2 = new conv_layer("conv2", 2, 208, 208, 16, 3, 1, 32, 1, LEAKY, true, false, NULL, NULL, 1);
    conv2->in(*max1_to_conv2);
    conv2->out(*conv2_to_max3);
    conv2->os(*os);        

    max3 = new max_layer("max3", 3, 208, 208, 32, 2, 2, false, NULL, NULL, 1); 
    max3->in(*conv2_to_max3);
    max3->out(*max3_to_conv4);
    max3->os(*os);
    
    conv4 = new conv_layer("conv4", 4, 104, 104, 32, 3, 1, 64, 1, LEAKY, true, false, NULL, NULL, 1);
    conv4->in(*max3_to_conv4);
    conv4->out(*conv4_to_max5);
    conv4->os(*os);

    max5 = new max_layer("max5", 5, 104, 104, 64, 2, 2, false, NULL, NULL, 1);
    max5->in(*conv4_to_max5);
    max5->out(*max5_to_conv6);
    max5->os(*os);        
    
    conv6 = new conv_layer("conv6", 6, 52, 52, 64, 3, 1, 128, 1, LEAKY, true, false, NULL, NULL, 1);
    conv6->in(*max5_to_conv6);
    conv6->out(*conv6_to_max7);
    conv6->os(*os);    

    max7 = new max_layer("max7", 7, 52, 52, 128, 2, 2, false, NULL, NULL, 1);
    max7->in(*conv6_to_max7);
    max7->out(*max7_to_conv8);	
    max7->os(*os);        

    conv8 = new conv_layer("conv8", 8, 26, 26, 128, 3, 1, 256, 1, LEAKY, true, false, NULL, NULL, 1);
    conv8->in(*max7_to_conv8);
    conv8->out(*conv8_to_max9);
    conv8->os(*os);        

    max9 = new max_layer("max9", 9, 26, 26, 256, 2,2, false, NULL, NULL, 1);
    max9->in(*conv8_to_max9);
    max9->out(*max9_to_conv10);
    max9->os(*os);        

    conv10 = new conv_layer("conv10", 10, 13, 13, 256, 3, 1, 512, 1, LEAKY, true, false, NULL, NULL, 1);
    conv10->in(*max9_to_conv10);
    conv10->out(*conv10_to_max11);
    conv10->os(*os);

    // !!! NOTE !!! this is the only max layer with stride=1
 
    max11 = new max_layer("max11", 11, 13, 13, 512, 2, 1, false, NULL, NULL,1);
    max11->in(*conv10_to_max11);
    max11->out(*max11_to_conv12);
    max11->os(*os);

    conv12 = new conv_layer("conv12", 12, 13, 13, 512, 3, 1, 1024, 1, LEAKY, true, false, NULL, NULL,1);
    conv12->in(*max11_to_conv12);
    conv12->out(*conv12_to_conv13);
    conv12->os(*os);

    conv13 = new conv_layer("conv13", 13, 13, 13, 1024, 3, 1, 512, 1, LEAKY, true, false, NULL, NULL,1);
    conv13->in(*conv12_to_conv13);
    conv13->out(*conv13_to_conv14);
    conv13->os(*os); 

    conv14 = new conv_layer("conv14", 14, 13, 13, 512, 1, 1, 425, 1, LINEAR, false, false, NULL, NULL, 1);
    conv14->in(*conv13_to_conv14);
    conv14->out(*conv14_to_region);
    conv14->os(*os);

    region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                           true, 1, 1, true, 0.6, true, 13, 13, 425, 1);
    region->in(*conv14_to_region);
    region->im_in(*reader_to_writer);
    region->im_w_in(*int_reader_to_writer); 
    region->im_h_in(*int2_reader_to_writer);
    region->im_name_in(*char_reader_to_writer);
    region->os(*os);
}

//testing
kpn_neuralnet_os_accel::kpn_neuralnet_os_accel(sc_module_name name, os_channel *os) : sc_module(name)
{
    cout << "in kpn_neuralnet_os" << endl;
    //strs images = {"../../darknet/data/dog.jpg", "../../darknet/data/horses.jpg"};
    strs images = {"../../darknet/data/dog.jpg"};
    //std::string cfgFile = "../../darknet/cfg/yolov2-tiny.cfg";
    //std::string weightFile = "../../darknet/yolov2-tiny.weights";
    //char *cfgFileC = new char[cfgFile.length() + 1];
    //strcpy(cfgFileC, cfgFile.c_str());
    //char *weightFileC = new char[weightFile.length() + 1];
    //strcpy(weightFileC, weightFile.c_str());
    //network *net = load_network(cfgFileC, weightFileC, 0);
    //cout << "instantiating os" << endl;
    //bool verbose = true;
    //os = new os_channel("os", 100, verbose);
    cout << "instantiating channels" << endl;
    reader_to_conv0 	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    reader_to_conv0->os(*os);
    conv0_to_max1   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv0_to_max1->os(*os);
    max1_to_conv2   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max1_to_conv2->os(*os);
    conv2_to_max3   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv2_to_max3->os(*os);
    max3_to_conv4   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max3_to_conv4->os(*os);
    conv4_to_max5   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv4_to_max5->os(*os);
    max5_to_conv6   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max5_to_conv6->os(*os);
    conv6_to_max7   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv6_to_max7->os(*os);
    max7_to_conv8   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max7_to_conv8->os(*os);
    conv8_to_max9   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv8_to_max9->os(*os);
    max9_to_conv10   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    max9_to_conv10->os(*os);
    conv10_to_max11   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv10_to_max11->os(*os);
    conv14_to_region   	= new os_sc_fifo<float>(BIGGEST_FIFO_SIZE);
    conv14_to_region->os(*os);

    reader_to_writer 	= new os_sc_fifo<float>(800 * 600 * 3 * 20);
    reader_to_writer->os(*os);
    int_reader_to_writer	= new os_sc_fifo<int>(20); // needed to send im.w and im.h
    int_reader_to_writer->os(*os);
    int2_reader_to_writer 	= new os_sc_fifo<int>(20);
    int2_reader_to_writer->os(*os); 
    char_reader_to_writer  	= new os_sc_fifo<string>(20);
    char_reader_to_writer->os(*os);
    
    // Here is where we will indicate the parameters for each layer. These can
    // be found in the cfg file for yolov2-tiny in the darknet folder.
    reader0 = new image_reader("image_reader", images, 1);
    reader0->out(*reader_to_conv0);
    reader0->im_out(*reader_to_writer);
    reader0->im_w_out(*int_reader_to_writer); 
    reader0->im_h_out(*int2_reader_to_writer);
    reader0->im_name_out(*char_reader_to_writer);
    reader0->os(*os);


    //name, layerIndex, filterSize, stride, numFilters, pad, activation, batchNormalize
    conv0 = new conv_layer("conv0", 0, 416, 416, 3, 3, 1, 16, 1,  LEAKY, true, false, NULL, NULL, 1);
    conv0->in(*reader_to_conv0);
    conv0->out(*conv0_to_max1);
    conv0->os(*os);
    
    max1 = new max_layer("max1", 1, 416, 416, 16, 2, 2, false, NULL, NULL, 1); 
    max1->in(*conv0_to_max1);
    max1->out(*max1_to_conv2);
    max1->os(*os);
    
    conv2 = new conv_layer("conv2", 2, 208, 208, 16, 3, 1, 32, 1, LEAKY, true, false, NULL, NULL, 1);
    conv2->in(*max1_to_conv2);
    conv2->out(*conv2_to_max3);
    conv2->os(*os);        

    max3 = new max_layer("max3", 3, 208, 208, 32, 2, 2, false, NULL, NULL, 1); 
    max3->in(*conv2_to_max3);
    max3->out(*max3_to_conv4);
    max3->os(*os);
    
    conv4 = new conv_layer("conv4", 4, 104, 104, 32, 3, 1, 64, 1, LEAKY, true, false, NULL, NULL, 1);
    conv4->in(*max3_to_conv4);
    conv4->out(*conv4_to_max5);
    conv4->os(*os);

    max5 = new max_layer("max5", 5, 104, 104, 64, 2, 2, false, NULL, NULL, 1);
    max5->in(*conv4_to_max5);
    max5->out(*max5_to_conv6);
    max5->os(*os);        
    
    conv6 = new conv_layer("conv6", 6, 52, 52, 64, 3, 1, 128, 1, LEAKY, true, false, NULL, NULL, 1);
    conv6->in(*max5_to_conv6);
    conv6->out(*conv6_to_max7);
    conv6->os(*os);    

    max7 = new max_layer("max7", 7, 52, 52, 128, 2, 2, false, NULL, NULL, 1);
    max7->in(*conv6_to_max7);
    max7->out(*max7_to_conv8);	
    max7->os(*os);        

    conv8 = new conv_layer("conv8", 8, 26, 26, 128, 3, 1, 256, 1, LEAKY, true, false, NULL, NULL, 1);
    conv8->in(*max7_to_conv8);
    conv8->out(*conv8_to_max9);
    conv8->os(*os);        

    max9 = new max_layer("max9", 9, 26, 26, 256, 2,2, false, NULL, NULL, 1);
    max9->in(*conv8_to_max9);
    max9->out(*max9_to_conv10);
    max9->os(*os);        

    conv10 = new conv_layer("conv10", 10, 13, 13, 256, 3, 1, 512, 1, LEAKY, true, false, NULL, NULL, 1);
    conv10->in(*max9_to_conv10);
    conv10->out(*conv10_to_max11);
    conv10->os(*os);

    // !!! NOTE !!! this is the only max layer with stride=1
    max11 = new max_layer("max11", 11, 13, 13, 512, 2, 1, false, NULL, NULL, 1);
    max11->in(*conv10_to_max11);
    max11->os(*os);

    conv14 = new conv_layer("conv14", 14, 13, 13, 512, 1, 1, 425, 1, LINEAR, false, false, NULL, NULL, 1);
    conv14->out(*conv14_to_region);
    conv14->os(*os);

    region = new region_layer("region", (float*)ANCHORS, true, 80, 4, 5, true, 0.2, false, 5,
                           true, 1, 1, true, 0.6, true, 13, 13, 425, 1);
    region->in(*conv14_to_region);
    region->im_in(*reader_to_writer);
    region->im_w_in(*int_reader_to_writer); 
    region->im_h_in(*int2_reader_to_writer);
    region->im_name_in(*char_reader_to_writer);
    region->os(*os);
}

kpn_neuralnet_accelerated::kpn_neuralnet_accelerated(sc_module_name name) : sc_module(name)
{
    bool verbose = false;
    os = new os_channel("os", 100, verbose);
    os_to_accel = new os_to_accel_fifo<float>(BIGGEST_FIFO_SIZE);
    os_to_accel->os(*os);
    accel_to_os = new accel_to_os_fifo<float>(BIGGEST_FIFO_SIZE);
    accel_to_os->os(*os);
    neuralnet = new kpn_neuralnet_os_accel("kpn_neuralnet_os", os);
    neuralnet->max11->out(*os_to_accel);
    neuralnet->conv14->in(*accel_to_os);
    accel = new accelerator("accelerator");
    accel->in(*os_to_accel);
    accel->out(*accel_to_os); 
}
