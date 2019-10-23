#ifndef ACCELERATOR_H
#define ACCELERATOR_H
#include "../kahn_process.h"
#include "../../darknet/src/convolutional_layer.h"
#include "HWBus.h"
#include "kpn_BusSlave.h"

class   accelerator : public kahn_process
{
    public:
    sc_fifo_in<float> in;
    sc_fifo_out<float> out;

    convolutional_layer l1, l2;
    accelerator(sc_module_name name);
    void process() override;
    void init() override;
};


class   accelerator_to_bus : public kahn_process
{
    public:
    
    sc_port<kpn_BusSlave_ifc> accel_to_bus; 

    convolutional_layer l1, l2;
    accelerator_to_bus(sc_module_name name);
    void process() override;
    void init() override;
};
#endif // ACCELERATOR_H
